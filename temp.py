# misun = ('m', 'i', 's', 'u', 'n')
# print(misun[:])
# print(misun[0:])
# print(misun[0:0])
# print(misun[0:5])
# print(len(misun))
# print(misun[0:len(misun)])
# print(misun[0:-1])


# import pickle, dill

# f_str = 'x + 2'
# print(f_str)
# f = lambda x: eval(f_str)
# print(f)
# print(f(3))

# dill.dumps(f)

# import tensorflow as tf
# import pickle, dill

# inputs = tf.keras.Input([None, None, 3], name='input')
# pickle.dumps(inputs)



def function(templist: list):
    templist=[1,2,3]

def function_altered(templist: list):
    templist[0]=[1,2,3]

misun=[]
function(misun)
print(misun)

jimin=[[],]
function_altered(jimin)
print(jimin)