#!/usr/bin/env python3
"""
Test script to verify chat template implementation works correctly
"""
import sys
import os

from transformers import AutoTokenizer
from utils import get_formatted_input_and_target

def test_chat_template(model_path):
    """Test the chat template implementation"""
    print(f"Testing with model: {model_path}")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Test 1: Normal chat messages (user + assistant)
    print("\n" + "=" * 80)
    print("TEST 1: Normal Chat Messages (user + assistant)")
    print("=" * 80)
    
    test_messages_chat = [
        {
            "role": "user",
            "content": "Hello, how are you?"
        },
        {
            "role": "assistant",
            "content": "I'm doing well, thank you! How can I help you today?"
        },
        {
            "role": "user",
            "content": "Can you explain what a transformer is?"
        },
        {
            "role": "assistant",
            "content": "A transformer is a deep learning architecture..."
        }
    ]
    
    print("\nMessages:")
    for msg in test_messages_chat:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    input_ids, target_ids = get_formatted_input_and_target(
        test_messages_chat,
        tokenizer,
        IGNORE_TOKEN_ID=-100,
        mask_prompt=True
    )
    
    if input_ids is None or target_ids is None:
        print("❌ ERROR: Function returned None!")
        return False
    
    print(f"\n✅ Success!")
    print(f"  Input IDs length: {len(input_ids)}")
    print(f"  Target IDs length: {len(target_ids)}")
    print(f"  Lengths match: {len(input_ids) == len(target_ids)}")
    
    masked_count = sum(1 for t in target_ids if t == -100)
    unmasked_count = len(target_ids) - masked_count
    print(f"  Masked tokens: {masked_count} ({masked_count/len(target_ids)*100:.1f}%)")
    print(f"  Unmasked tokens: {unmasked_count} ({unmasked_count/len(target_ids)*100:.1f}%)")
    
    print(f"\nFirst 50 tokens decoded:")
    decoded = tokenizer.decode(input_ids[:50])
    print(f"  {decoded[:150]}")
    
    # Test 2: Assistant-only messages (pure text continuation)
    print("\n" + "=" * 80)
    print("TEST 2: Assistant-Only Messages (pure text continuation)")
    print("=" * 80)
    
    test_messages_assistant_only = [
        {
            "role": "assistant",
            "content": "Once upon a time, there was a brave knight who lived in a castle."
        },
        {
            "role": "assistant",
            "content": "The knight had a magical sword that could defeat any enemy."
        }
    ]
    
    print("\nMessages:")
    for msg in test_messages_assistant_only:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    input_ids_asst, target_ids_asst = get_formatted_input_and_target(
        test_messages_assistant_only,
        tokenizer,
        IGNORE_TOKEN_ID=-100,
        mask_prompt=True
    )
    
    if input_ids_asst is None or target_ids_asst is None:
        print("❌ ERROR: Function returned None!")
        return False
    
    print(f"\n✅ Success!")
    print(f"  Input IDs length: {len(input_ids_asst)}")
    print(f"  Target IDs length: {len(target_ids_asst)}")
    print(f"  Lengths match: {len(input_ids_asst) == len(target_ids_asst)}")
    
    masked_count_asst = sum(1 for t in target_ids_asst if t == -100)
    unmasked_count_asst = len(target_ids_asst) - masked_count_asst
    print(f"  Masked tokens: {masked_count_asst} ({masked_count_asst/len(target_ids_asst)*100:.1f}%)")
    print(f"  Unmasked tokens: {unmasked_count_asst} ({unmasked_count_asst/len(target_ids_asst)*100:.1f}%)")
    
    print(f"\nFirst 50 tokens decoded:")
    decoded_asst = tokenizer.decode(input_ids_asst[:50])
    print(f"  {decoded_asst[:150]}")
    
    # Verify no chat template markers in assistant-only
    decoded_full = tokenizer.decode(input_ids_asst)
    has_chat_markers = any(marker in decoded_full for marker in ['<|im_start|>', '<|start_header_id|>', '[INST]'])
    if has_chat_markers:
        print("\n⚠️  WARNING: Chat template markers found in assistant-only mode!")
        print(f"  Decoded text: {decoded_full[:200]}")
    else:
        print("\n✅ No chat template markers (as expected for assistant-only)")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test chat template implementation")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model path to test"
    )
    
    args = parser.parse_args()
    
    try:
        success = test_chat_template(args.model)
        if success:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
