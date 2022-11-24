import tensorflow as tf

from tensor import Tensor

def test1():
    """Compare tensorflow and simplegrads simple gradient calculation"""
    
    # Tensorflow computations
    x1 = tf.Variable(3.4, name='x1')
    x2 = tf.Variable(2.6, name='x2')
    x3 = tf.Variable(5., name='x3')

    with tf.GradientTape() as tape:
        x4 = x1 + x2
        x5 = x3 * x4

    grad = tape.gradient(x5 ,{'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})
    for g in grad:
        print(f"Variable: {g} gradient: {grad[g].numpy()}")

    # Simplegrad computations
    t1 = Tensor(3.4)
    t2 = Tensor(2.6)
    t3 = Tensor(5)

    t4 = t1 + t2
    t5 = t3 * t4

    t5.backward()
    
    assert(t5.grad == grad['x5'].numpy())
    assert(t4.grad == grad['x4'].numpy())
    assert(t3.grad == grad['x3'].numpy())
    assert(t2.grad == grad['x2'].numpy())
    assert(t1.grad == grad['x1'].numpy())

    print("Test cases passed!")

def main():
    test1()

if __name__ == "__main__":
    main()