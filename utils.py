import os
import random
import torch
import torch.distributed as dist
import boto3


def get_formatted_input_and_target(messages, tokenizer, IGNORE_TOKEN_ID=-100, mask_prompt=True):
    """
    Convert messages to input_ids and target_ids using tokenizer's chat template.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tokenizer: HuggingFace tokenizer with chat template support
        IGNORE_TOKEN_ID: Token ID to use for masked positions (default: -100)
        mask_prompt: Whether to mask user prompts in targets (default: True)
    
    Returns:
        [input_ids, target_ids]: Lists of token IDs
    """
    try:
        # Check if all messages are assistant-only (pure text continuation scenario)
        all_assistant = all(msg.get('role') == 'assistant' for msg in messages)
        use_chat_template = not all_assistant
        
        # Generate input_ids
        if use_chat_template:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False
            )
        else:
            # Plain text tokenization for assistant-only messages
            input_ids = []
            if tokenizer.bos_token_id is not None:
                input_ids.append(tokenizer.bos_token_id)
            for message in messages:
                content = message.get('content', '')
                input_ids.extend(tokenizer.encode(content, add_special_tokens=False))
                if tokenizer.eos_token_id is not None:
                    input_ids.append(tokenizer.eos_token_id)
        
        # Generate target_ids with masking
        target_ids = []
        current_pos = 0
        
        for idx, message in enumerate(messages):
            # Determine token range for this message
            if use_chat_template:
                messages_so_far = messages[:idx+1]
                tokens_so_far = input_ids if idx == len(messages) - 1 else tokenizer.apply_chat_template(
                    messages_so_far,
                    tokenize=True,
                    add_generation_prompt=False
                )
                num_new_tokens = len(tokens_so_far) - current_pos
                tokens = tokens_so_far[current_pos:current_pos + num_new_tokens]
            else:
                # For plain text, calculate token positions manually
                content = message.get('content', '')
                content_tokens = tokenizer.encode(content, add_special_tokens=False)
                
                # Account for BOS token on first message
                if idx == 0 and tokenizer.bos_token_id is not None:
                    tokens = [tokenizer.bos_token_id] + content_tokens
                else:
                    tokens = content_tokens
                
                # Add EOS token
                if tokenizer.eos_token_id is not None:
                    tokens = tokens + [tokenizer.eos_token_id]
                
                num_new_tokens = len(tokens)
            
            # Determine if this message should be masked
            should_mask = _should_mask_message(message, mask_prompt)
            
            if should_mask:
                target_ids.extend([IGNORE_TOKEN_ID] * num_new_tokens)
            else:
                target_ids.extend(tokens if not use_chat_template else input_ids[current_pos:current_pos + num_new_tokens])
            
            current_pos += num_new_tokens
        
        # Ensure input_ids and target_ids have the same length
        assert len(input_ids) == len(target_ids), f"Length mismatch: {len(input_ids)} vs {len(target_ids)}"
        
        return [input_ids, target_ids]
    
    except Exception as e:
        print(f"Error in get_formatted_input_and_target: {e}")
        print(f"Messages: {messages}")
        return [None, None]


def _should_mask_message(message, mask_prompt):
    """Helper function to determine if a message should be masked"""
    role = message.get('role', '')
    
    # Check explicit mask flag
    if message.get('mask', 0) == 1:
        return True
    
    # Mask based on role
    if role == 'user':
        return mask_prompt
    elif role == 'system':
        return True
    elif role == 'assistant':
        return False
    else:
        print(f"Warning: Unknown role '{role}', masking tokens")
        return True


def get_examples_from_buffer_pad(buffer, seq_length, tokenizer, random_concat_ratio, IGNORE_TOKEN_ID=-100):
    all_input_ids_list, all_target_ids_list = [], []
    all_input_ids, all_target_ids = [], []

    for input_ids, target_ids in buffer:
        if len(input_ids) > seq_length - len(all_input_ids):
            input_ids = input_ids[-(seq_length - len(all_input_ids)):]
            target_ids = target_ids[-(seq_length - len(all_target_ids)):]
        if len(all_input_ids) > 0 and random.random() < random_concat_ratio:
            input_ids = input_ids[1:]
            target_ids = target_ids[1:]
        all_input_ids.extend(input_ids)
        all_target_ids.extend(target_ids)
        if len(all_input_ids) >= seq_length:
            assert len(all_input_ids) == seq_length, f"{len(all_input_ids)=}, {seq_length=}, {len(buffer)=}"
            all_input_ids_list.append(all_input_ids)
            all_target_ids_list.append(all_target_ids)
            all_input_ids, all_target_ids = [], []

    all_input_ids = all_input_ids + [tokenizer.pad_token_id for i in range(seq_length - len(all_input_ids))]
    all_target_ids = all_target_ids + [IGNORE_TOKEN_ID for i in range(seq_length - len(all_target_ids))]
    all_input_ids_list.append(all_input_ids)
    all_target_ids_list.append(all_target_ids)

    if len(all_input_ids) <= 0:
        return None
    return {
        "input_ids": torch.tensor(all_input_ids_list, dtype=torch.long),
        "labels": torch.tensor(all_target_ids_list, dtype=torch.long)
    }


def init_parallel_groups(ep_size=1):
    dist.init_process_group("nccl")
    world_size = int(os.getenv("WORLD_SIZE", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    ep_group = edp_group = None
    for i in range(0, world_size, ep_size):
        ranks = list(range(i, i + ep_size))
        group = dist.new_group(ranks)
        if local_rank in ranks:
            ep_group = group
    edp_group = None
    for i in range(ep_size):
        ranks = list(range(i, world_size, ep_size))
        group = dist.new_group(ranks)
        if local_rank in ranks:
            edp_group = group
    dist.all_reduce(torch.zeros(1, device="cuda"), group=ep_group)
    dist.all_reduce(torch.zeros(1, device="cuda"), group=edp_group)
    return world_size, local_rank, ep_group, edp_group


### AWS utils
def s3_upload(data_path, s3_path):
    s3 = boto3.client('s3')
    
    # Parse s3://bucket/key format
    parts = s3_path.replace('s3://', '').split('/')
    bucket = parts[0]
    key = os.path.join(*parts[1:], os.path.basename(data_path))

    s3.upload_file(
        data_path,  # filename
        bucket,     # bucket 
        key         # key
    )
    print(f"{data_path} is uploaded at s3://{bucket}/{key}")
