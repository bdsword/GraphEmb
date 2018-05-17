archs = ['x86_64_O0', 'x86_64_O1', 'x86_64_O2', 'x86_64_O3', 'arm', 'win']
compile_cmds = {
    'x86_64_O0': ['g++ -O0 -std=c++14 -w -o "{output}" "{src}"', 'g++ -O0 -std=c++11 -w -o "{output}" "{src}"', 'gcc -O0 -w -o "{output}" "{src}"', 'g++ -O0 -std=c++98 -w -o "{output}" "{src}"', 'g++ -O0 -std=c++17 -w -o "{output}" "{src}"', 'g++ -O0 -std=c++2a -w -o "{output}" "{src}"', 'g++ -O0 -std=c99 -w -o "{output}" "{src}"'],
    'x86_64_O1': ['g++ -O1 -std=c++14 -w -o "{output}" "{src}"', 'g++ -O1 -std=c++11 -w -o "{output}" "{src}"', 'gcc -O1 -w -o "{output}" "{src}"', 'g++ -O1 -std=c++98 -w -o "{output}" "{src}"', 'g++ -O1 -std=c++17 -w -o "{output}" "{src}"', 'g++ -O1 -std=c++2a -w -o "{output}" "{src}"', 'g++ -O1 -std=c99 -w -o "{output}" "{src}"'],
    'x86_64_O2': ['g++ -O2 -std=c++14 -w -o "{output}" "{src}"', 'g++ -O2 -std=c++11 -w -o "{output}" "{src}"', 'gcc -O2 -w -o "{output}" "{src}"', 'g++ -O2 -std=c++98 -w -o "{output}" "{src}"', 'g++ -O2 -std=c++17 -w -o "{output}" "{src}"', 'g++ -O2 -std=c++2a -w -o "{output}" "{src}"', 'g++ -O2 -std=c99 -w -o "{output}" "{src}"'],
    'x86_64_O3': ['g++ -O3 -std=c++14 -w -o "{output}" "{src}"', 'g++ -O3 -std=c++11 -w -o "{output}" "{src}"', 'gcc -O3 -w -o "{output}" "{src}"', 'g++ -O3 -std=c++98 -w -o "{output}" "{src}"', 'g++ -O3 -std=c++17 -w -o "{output}" "{src}"', 'g++ -O3 -std=c++2a -w -o "{output}" "{src}"', 'g++ -O3 -std=c99 -w -o "{output}" "{src}"'],
    'arm'      : ['arm-linux-gnueabi-g++ -std=c++14 -w -o "{output}" "{src}"'],
    'win'      : ['x86_64-w64-mingw32-g++-win32 -std=c++14 -w -o "{output}" "{src}"']
}

