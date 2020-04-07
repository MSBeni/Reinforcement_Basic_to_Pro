# ref: https://www.geeksforgeeks.org/enumerate-in-python/

# Enumerate() method adds a counter to an iterable and returns it in a form of enumerate object. This enumerate object
# can then be used directly in for loops or be converted into a list of tuples using list() method.

# Python program to illustrate
# enumerate function
l1 = ["eat", "sleep", "repeat"]
s1 = "geek"

# creating enumerate objects
obj1 = enumerate(l1)
obj2 = enumerate(s1)

print("Return type:", type(obj1))
print(list(enumerate(l1)))
print("\n")
print(list(enumerate(l1,10)))
print("\n")
# changing start index to 2 from 0
print(list(enumerate(s1, 2)))
print("\n")
######################################################################################################################

# Python program to illustrate
# enumerate function in loops
l1 = ["eat", "sleep", "repeat"]

# printing the tuples in object directly
for ele in enumerate(l1):
    print(ele)

# changing index and printing separately
for count, ele in enumerate(l1, 100):
    print(count, ele)


