
# coding: utf-8

# # Import Libraries

# In[1]:

import numpy as np
import tensorflow as tf
import helpers

tf.reset_default_graph()
sess = tf.InteractiveSession()

# In[2]:

#tf.__version__
helpers.pre_process_embeddings()
print("Loading numpy array")
#numpy_temp = np.random.rand(10000, 20)
#numpy_temp.astype(np.int32)
numpy_temp = np.load('data/embeddings_done.npy')
#numpy_temp = numpy_temp[()]
print("Loaded numpy array")
print(numpy_temp.shape[1])

# # Define model parameters sizes

# In[3]:

#PAD = 0
#EOS = 1
#vocab_size = 10
_, train_batch_, _, val_batch_, _, test_batch, vocab_size = helpers.m_load_data()
print vocab_size
PAD = vocab_size + 1
EOS = vocab_size + 2
input_embedding_size = numpy_temp.shape[1]
hidden_units = 20


# # Define input, output and weights

# In[4]:

inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs')
targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targets')


# In[5]:

#embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

#embeddings = tf.Variable(tf.constant(0.0, shape=[vocab_size, input_embedding_size]),
                #trainable=False, name="embeddings")   
    
init = tf.constant(numpy_temp, dtype=tf.float32)


embeddings = tf.get_variable('embeddings', trainable=False, initializer = init, dtype=tf.float32)


#embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, input_embedding_size])
#embedding_init = W.assign(embedding_placeholder)

# ...
#sess = tf.Session()

#sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})




# In[6]:

inputs_embedded = tf.nn.embedding_lookup(embeddings, inputs)


# In[7]:

cell = tf.contrib.rnn.LSTMCell(hidden_units)

outputs, final_state = tf.nn.dynamic_rnn(
    cell, inputs_embedded,
    dtype=tf.float32, time_major=True,
)


# In[8]:

logits = tf.contrib.layers.linear(outputs, vocab_size)

prediction = tf.argmax(logits, 2)


# In[9]:

#logits


# # Define loss

# In[10]:

# stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#     labels=tf.one_hot(encoder_targets, depth=vocab_size, dtype=tf.float32),
#     logits=encoder_logits,
# )

stepwise_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.transpose(targets),
    logits=logits
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)


# # Initialize variables

# In[11]:

sess.run(tf.global_variables_initializer())


# # Verify forward pass

# In[12]:

train_batch_, train_batch_length_ = helpers.batch(train_batch_)
#print('batch_encoded:\n' + str(train_batch_))

#pred_ = sess.run(encoder_prediction,
#    feed_dict={
#        encoder_inputs: train_batch_,
#        #encoder_inputs: train_batch_        
#    })
#print('encoder predictions:\n' + str(pred_))


# # Read and load data in time-major-form in batches

# In[13]:

#batch_size = 100
batch_size = 2

batches = helpers.read_and_load_data(train_batch_, batch_size)
#store = next(batches)
#print(store)
#print(helpers.batch(store))
#print(next(batches))


# In[14]:

def next_feed():
    batch = next(batches)
    #print type(batch)
    #print batch
    X_feed = helpers.batch(batch)[0]
    #print X_feed
    #print X_feed.shape
    return {
        inputs: X_feed[0:-1,:],
        targets: X_feed[1:,:]
    }



# In[15]:

#print(next_feed())
loss_track = []


# In[16]:

max_batches = 3000
batches_in_epoch = 100
#batches_in_epoch = 2

try:
    saver = tf.train.Saver()
    for batch in range(max_batches):
        #print "taking next et of files....."
        fd = next_feed()
        #print(fd)
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)
        
        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[inputs].T, predict_.T)):
                #print('  sample {}:'.format(i + 1))
                #print('    input     > {}'.format(inp))
                #print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            #print()
    save_path = saver.save(sess, "tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)           
    np.save("tmp/train_loss.npy",np.asarray(loss_track),allow_pickle=True)
    print("Losses saved")
except Exception as e:
    print('Training done')
    save_path = saver.save(sess, "tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    np.save("tmp/train_loss.npy",np.asarray(loss_track),allow_pickle=True)
    print("Losses saved")
finally:
    print(loss_track)


# In[17]:

#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.xlabel('Iterations')
plt.ylabel('Cross-entropy loss')
plt.title('Training loss sequence model')
plt.plot(loss_track)
plt.savefig("res/train_losses.png")
#print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))


# In[ ]:



