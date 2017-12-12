
import re



def check_arch(unknown_arch):
    available_archs = ['arm', 'x86']
    for arch in available_archs:
        if arch in unknown_arch:
            return arch
    return unknown_arch

def num_calls(code, arch):
    arch = check_arch(arch)
    arch_ins_pattern = {'arm': r'bl\s+.*$', 'x86': r'call\s+.*$'}
    counter = 0
    code = code.replace('\\l', '\n')
    instructions = re.findall(r'\s*0x[0-9a-f]{8}\s+(.*)', code)
    for ins in instructions:
        if len(re.findall(arch_ins_pattern[arch], ins)) != 0:
            counter += 1
    return counter


def num_instructions(code, arch):
    arch = check_arch(arch)
    code = code.replace('\\l', '\n')
    code = re.sub(r'^;.*$', r'', code)
    instructions = re.findall(r'\s*0x[0-9a-f]{8}\s+(.*)', code)
    return len(instructions)


def num_transfer(code, arch):
    arch = check_arch(arch)
    arch_ins_pattern = {'arm': r'(mov|mvn|ldr[b|h|sb|sh]{0,1}|str[b|h|sb|sh]{0,1}|ldm[ia|ib|da|db]{0,1}|stm[ia|ib|da|db]{0,1}|swp[b]{0,1}|push|pop)\s+.*', 'x86': r'(mov|xchg|cmpxchg|movz|movzx|movs|movsx|movsb|movsw|lea|pusha|popa|pushad|popad|lds|les|pushf|popf|lahf|sahf|in|out|push|pop)\s+.*'}
    counter = 0
    code = code.replace('\\l', '\n')
    instructions = re.findall(r'\s*0x[0-9a-f]{8}\s+(.*)', code)
    for ins in instructions:
        if len(re.findall(arch_ins_pattern[arch], ins)) != 0:
            counter += 1
    return counter


def num_arithmetic(code, arch):
    arch = check_arch(arch)
    arch_ins_pattern = {'arm': r'(add|adc|sub|sbc|rsb|rsc|and|eor|orr|bic|lsl|lsr|asr|ror|rrx)\s+.*', 'x86': r'(add|sub|inc|dec|imul|idiv|and|or|xor|not|neg|shl|shr)\s+.*'}
    counter = 0
    code = code.replace('\\l', '\n')
    instructions = re.findall(r'\s*0x[0-9a-f]{8}\s+(.*)', code)
    for ins in instructions:
        if len(re.findall(arch_ins_pattern[arch], ins)) != 0:
            counter += 1
    return counter

attributes_funcs = [num_calls, num_transfer, num_arithmetic, num_instructions]

