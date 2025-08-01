#!/usr/bin/env python3
"""
AIME数据集准确性分析脚本
比较每个问题的true_final_result和每个sample的答案
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

def is_correct_answer(final_result, true_final_result):
    """
    判断final_result是否与true_final_result匹配
    """
    return str(final_result).strip() == str(true_final_result).strip()

def analyze_aime_accuracy(input_json_path: str, output_path: str = None):
    """
    分析AIME数据集的准确性
    
    Args:
        input_json_path: 输入的JSON文件路径
        output_path: 输出结果文件路径（可选）
    """
    
    print("🔍 Loading AIME dataset...")
    
    # 检查文件是否存在
    if not os.path.exists(input_json_path):
        print(f"❌ Error: File {input_json_path} not found!")
        return
    
    # 加载数据
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Loaded {len(data)} problems")
    
    # 统计变量
    total_problems = len(data)
    correct_samples = {"sampling0": 0, "sampling1": 0, "sampling2": 0}
    total_samples = {"sampling0": 0, "sampling1": 0, "sampling2": 0}
    
    # 详细结果
    detailed_results = []
    
    print("\n" + "="*80)
    print("AIME ACCURACY ANALYSIS")
    print("="*80)
    
    for qid, sample in data.items():
        print(f"\n🔍 Analyzing {qid}...")
        
        # 获取基本信息
        original_data = sample.get("original_data", {})
        question_id = original_data.get("ID", "Unknown")
        year = original_data.get("Year", "Unknown")
        problem_num = original_data.get("Problem Number", "Unknown")
        question_text = original_data.get("Question", "")[:100] + "..." if len(original_data.get("Question", "")) > 100 else original_data.get("Question", "")
        
        true_final_result = sample.get("true_final_result", "")
        
        print(f"  📋 Question ID: {question_id}")
        print(f"  📅 Year: {year}, Problem: {problem_num}")
        print(f"  ✅ True Answer: {true_final_result}")
        print(f"  📝 Question: {question_text}")
        
        problem_result = {
            "qid": qid,
            "question_id": question_id,
            "year": year,
            "problem_num": problem_num,
            "true_answer": true_final_result,
            "question_text": original_data.get("Question", ""),
            "samples": {}
        }
        
        # 分析每个sample
        for sampling_id in ["sampling0", "sampling1", "sampling2"]:
            if sampling_id not in sample:
                print(f"    ⚠️ {sampling_id}: Not found")
                continue
            
            sampling_data = sample[sampling_id]
            final_result = sampling_data.get("final_result", "")
            is_correct = is_correct_answer(final_result, true_final_result)
            
            # 更新统计
            total_samples[sampling_id] += 1
            if is_correct:
                correct_samples[sampling_id] += 1
            
            # 记录结果
            problem_result["samples"][sampling_id] = {
                "final_result": final_result,
                "is_correct": is_correct,
                "whole_answer": sampling_data.get("whole_answer", "")[:200] + "..." if len(sampling_data.get("whole_answer", "")) > 200 else sampling_data.get("whole_answer", "")
            }
            
            status = "✅" if is_correct else "❌"
            print(f"    {status} {sampling_id}: {final_result} (Expected: {true_final_result})")
        
        detailed_results.append(problem_result)
    
    # 计算总体统计
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    overall_stats = {}
    for sampling_id in ["sampling0", "sampling1", "sampling2"]:
        if total_samples[sampling_id] > 0:
            accuracy = (correct_samples[sampling_id] / total_samples[sampling_id]) * 100
            overall_stats[sampling_id] = {
                "total_samples": total_samples[sampling_id],
                "correct_samples": correct_samples[sampling_id],
                "accuracy": accuracy
            }
            print(f"📊 {sampling_id}:")
            print(f"   Total: {total_samples[sampling_id]}")
            print(f"   Correct: {correct_samples[sampling_id]}")
            print(f"   Accuracy: {accuracy:.2f}%")
    
    # 计算平均准确率
    if overall_stats:
        avg_accuracy = sum(stats["accuracy"] for stats in overall_stats.values()) / len(overall_stats)
        print(f"\n📈 Average Accuracy: {avg_accuracy:.2f}%")
    
    # 保存详细结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        result_data = {
            "summary": {
                "total_problems": total_problems,
                "overall_stats": overall_stats,
                "average_accuracy": avg_accuracy if overall_stats else 0
            },
            "detailed_results": detailed_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Detailed results saved to: {output_path}")
    
    # 生成错误分析报告
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    error_analysis = {}
    for sampling_id in ["sampling0", "sampling1", "sampling2"]:
        error_analysis[sampling_id] = []
        
        for problem in detailed_results:
            if sampling_id in problem["samples"]:
                sample = problem["samples"][sampling_id]
                if not sample["is_correct"]:
                    error_analysis[sampling_id].append({
                        "qid": problem["qid"],
                        "question_id": problem["question_id"],
                        "true_answer": problem["true_answer"],
                        "wrong_answer": sample["final_result"],
                        "question_text": problem["question_text"][:100] + "..." if len(problem["question_text"]) > 100 else problem["question_text"]
                    })
    
    for sampling_id, errors in error_analysis.items():
        print(f"\n❌ {sampling_id} Errors ({len(errors)}):")
        for error in errors[:5]:  # 只显示前5个错误
            print(f"   {error['question_id']}: Expected {error['true_answer']}, Got {error['wrong_answer']}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")
    
    return result_data

def main():
    """主函数"""
    print("🎯 AIME Dataset Accuracy Analyzer")
    print("="*50)
    
    # 配置参数
    START_YEAR = 2022
    END_YEAR = 2023
    MIN_PROBLEM_NUM = 1
    MAX_PROBLEM_NUM = 5
    SAMPLE_SIZE = 10
    
    input_json = f"./logits/aime_experiment_{MIN_PROBLEM_NUM}-{MAX_PROBLEM_NUM}_{START_YEAR}-{END_YEAR}_sample{SAMPLE_SIZE}.json"
    output_results = f"./output/aime_accuracy_analysis_{MIN_PROBLEM_NUM}-{MAX_PROBLEM_NUM}_{START_YEAR}-{END_YEAR}_sample{SAMPLE_SIZE}.json"
    
    # 运行分析
    results = analyze_aime_accuracy(input_json, output_results)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Results saved to: {output_results}")

if __name__ == "__main__":
    main() 