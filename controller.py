import sys
#import serial

mode = sys.argv[1]
action = sys.argv[2]

print(mode, action)
sys.stdout.flush()
#ser = serial.Serial('/dev/myUSB', 9600)

#if x == 'f':
#    ser.write('6')
#elif x== 'l':
#    ser.write('L')
#elif x== 'r':
#    ser.write('R')
#elif x== 'c':
#    ser.write('Z')
