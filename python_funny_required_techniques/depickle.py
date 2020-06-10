import pickle

pickle_in = open("dict.picle", "rb")
input_dict = pickle.load(pickle_in)
print(input_dict)
print(input_dict[1])