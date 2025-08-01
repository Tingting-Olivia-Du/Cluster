# -*- coding: utf-8 -*-
"""
Complete JSONL processor with all required functions
Fixed for actual AIME data structure
"""

import json
import os
import gc
import requests
import re
import time
from pathlib import Path
from typing import Iterator, Dict, Any

# # ✅ API配置
API_KEY = "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"
MODEL = "gpt-4.1"

# ✅ 构建错误分析提示词
def build_error_prompt(question, true_whole_answer, sample_whole_answer):
    """构建用于错误分析的提示词"""
    return f"""
Here is a math question, its correct answer, and a sample answer that may contain mistakes.

【question】:
{question}

【Correct Answer】:
{true_whole_answer}

【Incorrect Answer】:
{sample_whole_answer}

Please help me:
1. Identify the earliest mistake in the incorrect answer and provide the complete sentence from that point.
2. Briefly explain why it is incorrect.
3. Find the fix sentence in correct answer that and fix the error.
4. Briefly explain why it can fix the error.

Please output in the following JSON format:
{{
  "first_error_sentence": "<sentence>",
  "error_reason": "<brief explanation>",
  "fix_sentence": "<sentence>",
  "fix_reason": "<brief explanation>"
}}
"""

# ✅ 调用GPT API
def call_custom_gpt_api(prompt):
    """调用OpenAI API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a meticulous and precise comparer."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30  # 添加超时
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}, {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        raise Exception("API request timeout")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request error: {str(e)}")

# ✅ 查找句子在token序列中的位置
def find_sentence_span_indices_robust(fragment, token_probs):
    """
    返回 fragment 在 token_probs 中匹配到的 token 范围: (begin_index, end_index)
    使用去除空白字符的方式匹配
    """
    if not fragment or not token_probs:
        return -1, -1

    fragment_clean = re.sub(r"\s+", "", fragment)
    tokens = [entry["token"] for entry in token_probs]
    decoded_text = "".join(tokens)
    decoded_text_clean = re.sub(r"\s+", "", decoded_text)

    char_start_idx = decoded_text_clean.find(fragment_clean)
    if char_start_idx == -1:
        return -1, -1

    cumulative_len = 0
    begin_index = -1

    for idx, entry in enumerate(token_probs):
        token_clean = re.sub(r"\s+", "", entry["token"])
        prev_len = cumulative_len
        cumulative_len += len(token_clean)

        if begin_index == -1 and cumulative_len > char_start_idx:
            begin_index = idx
        if cumulative_len >= char_start_idx + len(fragment_clean):
            end_index = idx
            return begin_index, end_index

    return begin_index, len(token_probs) - 1  # fallback

# ✅ 简化的答案判断函数（适配AIME数据集）
def is_correct_answer(final_result, true_final_result):
    """
    判断final_result是否与true_final_result匹配
    适配AIME数据集，直接进行字符串比较
    """
    return str(final_result).strip() == str(true_final_result).strip()

class JSONLProcessor:
    """
    高效的JSONL处理器，支持内存管理和进度跟踪
    适配AIME数据集的实际结构
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        self.api_key = api_key
        self.model = model
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None

    def convert_json_to_jsonl(self, input_path: str, output_path: str,
                             chunk_size: int = 1000):
        """
        将大JSON文件转换为JSONL，支持分块处理
        """
        print(f"🔄 Converting {input_path} to JSONL format...")

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 检查文件大小
        file_size = os.path.getsize(input_path)
        print(f"📊 Input file size: {file_size / (1024**3):.2f} GB")

        with open(input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        total_items = len(data)
        print(f"📊 Total items to convert: {total_items}")

        with open(output_path, 'w', encoding='utf-8') as outfile:
            for i, (qid, sample) in enumerate(data.items()):
                line_data = {
                    "qid": qid,
                    "data": sample
                }
                outfile.write(json.dumps(line_data, ensure_ascii=False) + '\n')

                if (i + 1) % chunk_size == 0:
                    print(f"📈 Converted {i + 1}/{total_items} items...")
                    # 强制刷新到磁盘
                    outfile.flush()

        print(f"✅ Conversion complete! Saved to {output_path}")

        # 清理内存
        del data
        gc.collect()

    def process_jsonl_file(self, jsonl_path: str, output_path: str,
                          batch_size: int = 10, save_interval: int = 20):
        """
        流式处理JSONL文件，支持批处理和定期保存
        """
        self.start_time = time.time()
        results = {}

        # 如果输出文件已存在，加载已处理的结果
        if os.path.exists(output_path):
            print("📂 Loading existing results...")
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    self.processed_count = len(results)
                    print(f"📊 Loaded {self.processed_count} existing results")
            except (json.JSONDecodeError, FileNotFoundError):
                print("⚠️ Could not load existing results, starting fresh")
                results = {}

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            batch = []
            line_count = 0

            for line in f:
                try:
                    line_data = json.loads(line.strip())
                    qid = line_data["qid"]

                    # 跳过已处理的项目
                    if qid in results:
                        print(f"⏭️ Skipping already processed: {qid}")
                        continue

                    batch.append((qid, line_data["data"]))
                    line_count += 1

                    # 处理批次
                    if len(batch) >= batch_size:
                        self._process_batch(batch, results)
                        batch = []

                        # 定期保存和清理内存
                        if line_count % save_interval == 0:
                            self._save_results(results, output_path)
                            gc.collect()
                            self._print_progress()

                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error in line: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ Unexpected error processing line: {e}")
                    continue

            # 处理剩余的项目
            if batch:
                self._process_batch(batch, results)

        # 最终保存
        self._save_results(results, output_path)
        self._print_final_stats()

        return results

    def _process_batch(self, batch: list, results: dict):
        """处理一个批次的数据"""
        for qid, sample in batch:
            try:
                print(f"🔍 Processing {qid}...")
                result = self._process_single_sample(qid, sample)
                if result:
                    results[qid] = result
                    self.processed_count += 1
                    print(f"✅ Successfully processed {qid}")
                else:
                    print(f"⚠️ No valid result for {qid}")

            except Exception as e:
                print(f"⚠️ Error processing {qid}: {str(e)}")
                self.error_count += 1
                continue

    def _process_single_sample(self, qid: str, sample: dict) -> dict:
        """处理单个样本 - 适配实际AIME数据结构"""
        try:
            # 获取基本信息 - 适配实际数据结构
            question = sample.get("full_question", "")
            true_final_result = sample.get("true_final_result", "")

            if not question or not true_final_result:
                print(f"⚠️ Missing full_question or true_final_result for {qid}")
                return None

            print(f"  📋 Question: {question[:100]}...")
            print(f"  ✅ True answer: {true_final_result}")

            # 找到正样本
            correct_sampling_id = None
            correct_sample_answer = None

            for sampling_id in ["sampling0", "sampling1", "sampling2"]:
                if sampling_id not in sample:
                    continue
                    
                sampling_data = sample[sampling_id]
                final_result = sampling_data.get("final_result", "")
                
                # 使用简化的判断逻辑
                if is_correct_answer(final_result, true_final_result):
                    correct_sampling_id = sampling_id
                    correct_sample_answer = sampling_data.get("whole_answer", "")
                    print(f"  ✅ Found correct sample: {sampling_id} (result: {final_result})")
                    break

            if correct_sample_answer is None:
                print(f"⚠️ No correct sampling found for {qid}")
                # 打印调试信息
                for sampling_id in ["sampling0", "sampling1", "sampling2"]:
                    if sampling_id in sample:
                        final_result = sample[sampling_id].get("final_result", "")
                        print(f"    {sampling_id}: {final_result} (expected: {true_final_result})")
                return None

            sample_results = {}

            # 处理负样本
            for sampling_id in ["sampling0", "sampling1", "sampling2"]:
                if sampling_id not in sample:
                    continue
                    
                sampling = sample[sampling_id]
                final_result = sampling.get("final_result", "")

                # 跳过正样本
                if is_correct_answer(final_result, true_final_result):
                    print(f"  ⏭️ Skipping correct sample: {sampling_id}")
                    continue

                incorrect_sample_answer = sampling.get("whole_answer", "")
                if not incorrect_sample_answer:
                    continue

                try:
                    print(f"  🔍 Analyzing incorrect sample: {sampling_id} (result: {final_result})")

                    # 调用API
                    prompt = build_error_prompt(question, correct_sample_answer, incorrect_sample_answer)
                    output = call_custom_gpt_api(prompt)

                    # 解析结果
                    output = output.strip().strip("```")
                    if output.startswith("json"):
                        output = output[4:].strip()

                    output_json = json.loads(output)

                    # 查找token索引
                    error_sentence = output_json.get("first_error_sentence", "")
                    fix_sentence = output_json.get("fix_sentence", "")

                    error_token_probs = sampling.get("token_probs", [])
                    correct_token_probs = sample[correct_sampling_id].get("token_probs", [])

                    error_begin_idx, error_end_idx = find_sentence_span_indices_robust(
                        error_sentence, error_token_probs
                    )
                    fix_begin_idx, fix_end_idx = find_sentence_span_indices_robust(
                        fix_sentence, correct_token_probs
                    )

                    sample_results[sampling_id] = {
                        "first_error_sentence": error_sentence,
                        "error_reason": output_json.get("error_reason", ""),
                        "fix_sentence": fix_sentence,
                        "fix_reason": output_json.get("fix_reason", ""),
                        "correct_sampling_id": correct_sampling_id,
                        "error_token_begin_index": error_begin_idx,
                        "error_token_end_index": error_end_idx,
                        "fix_token_begin_index": fix_begin_idx,
                        "fix_token_end_index": fix_end_idx
                    }

                    print(f"  ✅ Successfully analyzed {sampling_id}")

                except json.JSONDecodeError as e:
                    print(f"  ⚠️ JSON decode error for {sampling_id}: {e}")
                    continue
                except Exception as e:
                    print(f"  ⚠️ Error analyzing {sampling_id}: {e}")
                    continue

            return sample_results if sample_results else None

        except Exception as e:
            print(f"⚠️ Error in _process_single_sample for {qid}: {e}")
            return None

    def _save_results(self, results: dict, output_path: str):
        """保存结果到文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Saved {len(results)} results to {output_path}")
        except Exception as e:
            print(f"⚠️ Error saving results: {e}")

    def _print_progress(self):
        """打印进度信息"""
        elapsed = time.time() - self.start_time
        speed = self.processed_count / elapsed if elapsed > 0 else 0
        print(f"📊 Progress: {self.processed_count} processed, {self.error_count} errors, "
              f"{speed:.2f} items/sec, {elapsed:.1f}s elapsed")

    def _print_final_stats(self):
        """打印最终统计信息"""
        elapsed = time.time() - self.start_time
        print(f"\n🎉 Processing complete!")
        print(f"📊 Total processed: {self.processed_count}")
        print(f"⚠️ Total errors: {self.error_count}")
        print(f"⏱️ Total time: {elapsed:.1f}s")
        if elapsed > 0:
            print(f"🚀 Average speed: {self.processed_count / elapsed:.2f} items/sec")

# ✅ 主函数
def main():
    # 路径配置 - 适配AIME数据集
    range_tag = "11-30"
    START_YEAR = 2022
    END_YEAR = 2023
    MIN_PROBLEM_NUM = 1
    MAX_PROBLEM_NUM = 5
    SAMPLE_SIZE = 10
    
    input_json = f"./logits/aime_experiment_{MIN_PROBLEM_NUM}-{MAX_PROBLEM_NUM}_{START_YEAR}-{END_YEAR}_sample{SAMPLE_SIZE}.json"
    output_jsonl = f"./logits/aime_experiment_{MIN_PROBLEM_NUM}-{MAX_PROBLEM_NUM}_{START_YEAR}-{END_YEAR}_sample{SAMPLE_SIZE}.jsonl"
    output_results = f"./error_fix_index/aime_experiment_{MIN_PROBLEM_NUM}-{MAX_PROBLEM_NUM}_{START_YEAR}-{END_YEAR}_sample{SAMPLE_SIZE}_index.json"

    # 创建处理器
    processor = JSONLProcessor(API_KEY, MODEL)

    # 步骤1: 转换为JSONL（如果还没有转换）
    if not os.path.exists(output_jsonl):
        processor.convert_json_to_jsonl(input_json, output_jsonl)
    else:
        print(f"📂 JSONL file already exists: {output_jsonl}")

    # 步骤2: 处理JSONL文件
    print("\n🚀 Starting JSONL processing...")
    results = processor.process_jsonl_file(
        jsonl_path=output_jsonl,
        output_path=output_results,
        batch_size=1,      # 设置为1以便调试
        save_interval=5    # 每5个项目保存一次
    )

    print(f"\n✅ All processing complete! Results saved to {output_results}")

if __name__ == "__main__":
    main() 