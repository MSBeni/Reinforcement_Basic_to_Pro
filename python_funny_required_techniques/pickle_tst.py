import pickle

pickle_in = {1:"1", 2:"a", 15:"12"}
pickle_out = open("dict.picle", "wb")
pickle.dump(pickle_in, pickle_out)
pickle_out.close()