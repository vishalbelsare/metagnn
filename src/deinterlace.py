import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Splits an interleaved FASTQ into first and second read files')
    parser.add_argument('fastq')
    parser.add_argument('first')
    parser.add_argument('second')
    args = parser.parse_args()

    print(args)
    fastq = open(args.fastq, "r")
    pipeForward = open(args.first, "w+")
    pipeReverse = open(args.second, "w+")

    i = 0
    forward = True
    for line in fastq:
        if forward:
            pipeForward.write(line)
        else:
            pipeReverse.write(line)
        i += 1
        if i == 4:
            forward = not forward
            i = 0
    
    fastq.close()
    pipeForward.close()
    pipeReverse.close()

if __name__ == '__main__':
    main()
