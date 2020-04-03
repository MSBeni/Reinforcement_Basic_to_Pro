import argparse
import math

parser = argparse.ArgumentParser(description='Process of calculating the volume')
parser.add_argument('-r', '--radius', type=int, metavar='', required=True, help='add the radius of the cube')    # working with the flag -r
parser.add_argument('-H', '--height', type=int, metavar='', required=True, help='add the height of the cube')    # working with the flag -H
# parser.add_argument('radius', type=int, help='add the radius of the cube')
# parser.add_argument('height', type=int, help='add the height of the cube')

group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quite', action='store_true', help='print quite')
group.add_argument('-v', '--verbose', action='store_true', help='print verbose')

args = parser.parse_args()


def cube_vol_cal(r,h):
    return 2*math.pi*r*r*h

if __name__ == "__main__":
    volume = cube_vol_cal(args.radius, args.height)
    if args.quite:
        print(volume)
    elif args.verbose:
        print("volume of the cylinder with radius {} and height {}, is {}".format(args.radius, args.height, volume))
    else:
        print('volume of the cylinder is: {}'.format(volume))
