#!/usr/bin/env python3
"""
DLL 导出符号验证脚本

该脚本自动验证 boat.dll 中的导出符号，确保所有关键函数都已正确导出。
支持使用 objdump (GNU工具链) 或 dumpbin (MSVC工具链) 进行验证。
"""

import subprocess
import sys
import os
import re
from typing import List, Set, Tuple, Optional

class DLLExportVerifier:
    def __init__(self, dll_path: str, use_dumpbin: bool = False):
        """
        初始化验证器

        Args:
            dll_path: DLL文件路径
            use_dumpbin: 是否使用 dumpbin (默认为 False，使用 objdump)
        """
        self.dll_path = os.path.abspath(dll_path)
        self.use_dumpbin = use_dumpbin
        self.exported_functions: Set[str] = set()

        # 关键函数列表 - 这些函数必须被导出
        self.critical_functions = [
            # 注意力层函数
            "boat_attention_layer_forward",
            "boat_attention_layer_backward",
            "boat_attention_layer_update",

            # 归一化层函数
            "boat_norm_layer_forward",
            "boat_norm_layer_backward",
            "boat_norm_layer_update",

            # 全连接层函数
            "boat_dense_layer_backward",
            "boat_dense_layer_update",

            # 卷积层函数 (getter函数)
            "boat_conv_layer_get_bias",
            "boat_conv_layer_get_grad_bias",
            "boat_conv_layer_get_grad_weight",
            "boat_conv_layer_get_weight",

            # 注意力层简单函数 (getter/setter)
            "boat_attention_get_bias_k",
            "boat_attention_get_bias_o",
            "boat_attention_get_bias_q",
            "boat_attention_get_bias_v",
            "boat_attention_get_grad_bias_k",
            "boat_attention_get_grad_bias_o",
            "boat_attention_get_grad_bias_q",
            "boat_attention_get_grad_bias_v",
            "boat_attention_get_grad_weight_k",
            "boat_attention_get_grad_weight_o",
            "boat_attention_get_grad_weight_q",
            "boat_attention_get_grad_weight_v",
            "boat_attention_get_weight_k",
            "boat_attention_get_weight_o",
            "boat_attention_get_weight_q",
            "boat_attention_get_weight_v",
            "boat_attention_set_causal",
            "boat_attention_set_dropout",
        ]

    def run_command(self, cmd: List[str]) -> Tuple[str, str, int]:
        """
        运行命令并返回输出

        Returns:
            (stdout, stderr, returncode)
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            return result.stdout, result.stderr, result.returncode
        except FileNotFoundError:
            print(f"错误: 命令未找到: {' '.join(cmd)}")
            return "", "", 1

    def extract_exports_with_objdump(self) -> bool:
        """
        使用 objdump -p 提取导出函数

        Returns:
            True if successful, False otherwise
        """
        cmd = ["objdump", "-p", self.dll_path]
        stdout, stderr, returncode = self.run_command(cmd)

        if returncode != 0:
            print(f"objdump 失败: {stderr}")
            return False

        # 解析导出函数名
        # objdump 输出中，导出函数名行格式为:
        # [   0] +base[   1]  0000 boat_adagrad_optimizer_create
        # 或者:
        # [   0] +base[   1] 0001e660 Export RVA
        # 我们需要匹配第一种格式（有函数名）
        pattern1 = r'^\s*\[\s*\d+\]\s*\+base\[\s*\d+\]\s+[0-9a-f]+\s+([a-zA-Z0-9_]+)$'
        # 也匹配没有十六进制数字的格式
        pattern2 = r'^\s*\[\s*\d+\]\s*\+base\[\s*\d+\]\s+([a-zA-Z0-9_]+)$'

        for line in stdout.split('\n'):
            line = line.rstrip()
            match = re.match(pattern1, line)
            if not match:
                match = re.match(pattern2, line)

            if match:
                func_name = match.group(1)
                if func_name.startswith('boat_'):
                    self.exported_functions.add(func_name)
                else:
                    # 可能是装饰名，尝试提取 boat_ 前缀
                    if 'boat_' in func_name:
                        idx = func_name.find('boat_')
                        self.exported_functions.add(func_name[idx:])

        # 如果上述模式没有匹配到，尝试另一种方法：查找包含 "boat_" 的行
        if not self.exported_functions:
            for line in stdout.split('\n'):
                if 'boat_' in line:
                    # 提取 boat_ 开头的单词
                    words = line.split()
                    for word in words:
                        if word.startswith('boat_'):
                            self.exported_functions.add(word)
                            break
                        elif 'boat_' in word:
                            idx = word.find('boat_')
                            self.exported_functions.add(word[idx:])
                            break

        return True

    def extract_exports_with_dumpbin(self) -> bool:
        """
        使用 dumpbin /exports 提取导出函数

        Returns:
            True if successful, False otherwise
        """
        cmd = ["dumpbin", "/exports", self.dll_path]
        stdout, stderr, returncode = self.run_command(cmd)

        if returncode != 0:
            print(f"dumpbin 失败: {stderr}")
            return False

        # dumpbin 输出中，导出函数表在 "ordinal hint RVA      name" 行之后
        lines = stdout.split('\n')
        in_export_table = False

        for line in lines:
            line = line.rstrip()

            # 检测导出表开始
            if "ordinal" in line.lower() and "hint" in line.lower() and "RVA" in line.lower() and "name" in line.lower():
                in_export_table = True
                continue

            if in_export_table:
                # 空行表示表结束
                if not line.strip():
                    break

                # 解析行：例如 "    1    0 0001E660 boat_adagrad_optimizer_create"
                parts = line.strip().split()
                if len(parts) >= 4:
                    # 最后一个部分可能是函数名（可能有装饰名）
                    func_name = parts[-1]

                    # 去除可能的装饰（如 __cdecl 前缀）
                    if func_name.startswith('_'):
                        # 可能是装饰名，尝试提取
                        # 简化处理：只接受 boat_ 开头的函数
                        if 'boat_' in func_name:
                            # 查找 boat_ 的位置
                            idx = func_name.find('boat_')
                            if idx > 0:
                                func_name = func_name[idx:]

                    if func_name.startswith('boat_'):
                        self.exported_functions.add(func_name)

        return True

    def extract_exports(self) -> bool:
        """
        提取导出函数

        Returns:
            True if successful, False otherwise
        """
        if self.use_dumpbin:
            print(f"使用 dumpbin 提取导出函数: {self.dll_path}")
            return self.extract_exports_with_dumpbin()
        else:
            print(f"使用 objdump 提取导出函数: {self.dll_path}")
            return self.extract_exports_with_objdump()

    def verify_critical_functions(self) -> Tuple[bool, List[str], List[str]]:
        """
        验证关键函数是否已导出

        Returns:
            (success, missing_functions, found_functions)
        """
        missing = []
        found = []

        for func in self.critical_functions:
            if func in self.exported_functions:
                found.append(func)
            else:
                missing.append(func)

        success = len(missing) == 0
        return success, missing, found

    def run_verification(self) -> bool:
        """
        运行完整验证流程

        Returns:
            True if all critical functions are exported, False otherwise
        """
        print(f"=== DLL 导出符号验证 ===\nDLL: {self.dll_path}")

        # 提取导出函数
        if not self.extract_exports():
            print("错误: 无法提取导出函数")
            return False

        print(f"找到 {len(self.exported_functions)} 个导出函数")

        # 验证关键函数
        success, missing, found = self.verify_critical_functions()

        print(f"\n关键函数验证结果:")
        print(f"  已找到: {len(found)}/{len(self.critical_functions)}")
        print(f"  缺失: {len(missing)}/{len(self.critical_functions)}")

        if found:
            print(f"\n已找到的关键函数:")
            for func in sorted(found):
                print(f"  [OK] {func}")

        if missing:
            print(f"\n缺失的关键函数:")
            for func in sorted(missing):
                print(f"  [MISSING] {func}")

        if success:
            print(f"\n[SUCCESS] 验证通过: 所有关键函数都已导出")
        else:
            print(f"\n[FAILURE] 验证失败: {len(missing)} 个关键函数缺失")

        return success

    def get_export_count(self) -> int:
        """返回导出的函数总数"""
        return len(self.exported_functions)

    def list_all_exports(self, limit: int = 50) -> List[str]:
        """列出所有导出函数（可选限制数量）"""
        exports = sorted(self.exported_functions)
        if limit > 0 and len(exports) > limit:
            return exports[:limit]
        return exports


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DLL 导出符号验证脚本')
    parser.add_argument('dll_path', help='DLL文件路径')
    parser.add_argument('--dumpbin', action='store_true',
                       help='使用 dumpbin 而不是 objdump')
    parser.add_argument('--list', action='store_true',
                       help='列出所有导出函数')
    parser.add_argument('--limit', type=int, default=50,
                       help='列出导出函数时的数量限制')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查，不验证关键函数')

    args = parser.parse_args()

    # 检查DLL文件是否存在
    if not os.path.exists(args.dll_path):
        print(f"错误: DLL文件不存在: {args.dll_path}")
        return 1

    # 创建验证器
    verifier = DLLExportVerifier(args.dll_path, use_dumpbin=args.dumpbin)

    # 提取导出函数
    if not verifier.extract_exports():
        return 1

    # 如果指定了 --list，则列出导出函数
    if args.list:
        exports = verifier.list_all_exports(args.limit)
        print(f"导出函数 (共 {verifier.get_export_count()} 个):")
        for i, func in enumerate(exports, 1):
            print(f"{i:4d}: {func}")
        if args.limit > 0 and verifier.get_export_count() > args.limit:
            print(f"... 还有 {verifier.get_export_count() - args.limit} 个函数未显示")
        return 0

    # 运行验证
    success = verifier.run_verification()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())