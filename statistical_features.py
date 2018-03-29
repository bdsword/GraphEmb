
import re


def check_arch(unknown_arch):
    arch_maps = {'arm': 'arm',
                 'win': 'x86',
                 'x86_64_O0': 'x86',
                 'x86_64_O1': 'x86',
                 'x86_64_O2': 'x86',
                 'x86_64_O3': 'x86'}
    if unknown_arch in arch_maps:
        return arch_maps[unknown_arch]
    else:
        return unknown_arch


def num_calls(code, arch):
    arch = check_arch(arch)
    arch_ins_pattern = {'arm': r'bl\s+.*$', 'x86': r'call\s+.*$'}
    counter = 0
    code = code.replace('\\l', '\n')
    instructions = re.findall(r'(.*)(?<!:)\n', code)
    for ins in instructions:
        if len(re.findall(arch_ins_pattern[arch], ins)) != 0:
            counter += 1
    return counter * 66.22


def num_instructions(code, arch):
    arch = check_arch(arch)
    code = code.replace('\\l', '\n')
    code = re.sub(r'^;.*$', r'', code)
    instructions = re.findall(r'(.*)(?<!:)\n', code)
    return len(instructions) * 41.37


def num_transfer(code, arch):
    arch = check_arch(arch)
    arch_ins_pattern = {'arm': r'(mov|mvn|ldr[b|h|sb|sh]{0,1}|str[b|h|sb|sh]{0,1}|ldm[ia|ib|da|db]{0,1}|stm[ia|ib|da|db]{0,1}|swp[b]{0,1}|push|pop)\s+.*', 'x86': r'(mov|xchg|cmpxchg|movz|movzx|movs|movsx|movsb|movsw|lea|pusha|popa|pushad|popad|lds|les|pushf|popf|lahf|sahf|in|out|push|pop)\s+.*'}
    counter = 0
    code = code.replace('\\l', '\n')
    instructions = re.findall(r'(.*)(?<!:)\n', code)
    for ins in instructions:
        if len(re.findall(arch_ins_pattern[arch], ins)) != 0:
            counter += 1
    return counter * 6.54


def num_arithmetic(code, arch):
    arch = check_arch(arch)
    arch_ins_pattern = {'arm': r'(add|adc|sub|sbc|rsb|rsc|and|eor|orr|bic|lsl|lsr|asr|ror|rrx)\s+.*', 'x86': r'(add|sub|inc|dec|imul|idiv|and|or|xor|not|neg|shl|shr)\s+.*'}
    counter = 0
    code = code.replace('\\l', '\n')
    instructions = re.findall(r'(.*)(?<!:)\n', code)
    for ins in instructions:
        if len(re.findall(arch_ins_pattern[arch], ins)) != 0:
            counter += 1
    return counter * 55.65


def num_string(code, arch):
    arch = check_arch(arch)
    arch_ins_pattern = {'arm': r'.*;\s*".*"', 'x86': r'.*;\s*".*"'}
    counter = 0
    code = code.replace('\\l', '\n')
    instructions = re.findall(r'(.*)(?<!:)\n', code)
    for ins in instructions:
        if len(re.findall(arch_ins_pattern[arch], ins)) != 0:
            counter += 1
    return counter * 11.998


def num_numeric_constant(code, arch):
    arch = check_arch(arch)
    arch_ins_pattern = {'arm': r'[\+\-\*\/\s](\d|\d+h)[\+\-\*\/\s\n]', 'x86': r'[\+\-\*\/\s](\d|\d+h)[\+\-\*\/\s\n]'}
    counter = 0
    code = code.replace('\\l', '\n')
    instructions = re.findall(r'(.*)(?<!:)\n', code)
    for ins in instructions:
        if len(re.findall(arch_ins_pattern[arch], ins)) != 0:
            counter += 1
    return counter * 15.382


statistical_features = [num_calls, num_transfer, num_arithmetic, num_instructions, num_string, num_numeric_constant]

