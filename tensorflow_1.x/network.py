import tensorflow as tf
import numpy as np
import os
import sys


SCOPE = 'rendezvous'


class Rendezvous(object):
    def __init__(self, model='rendezvous', basename="resnet18", num_tool=6, num_verb=10, num_target=15, num_triplet=100, layer_size=8, num_heads=4, d_model=128, use_ln=False, hr_output=False):
        super(Rendezvous, self).__init__()
        self.encoder = Encoder(basename, num_tool, num_verb, num_target, num_triplet, hr_output=hr_output)
        self.decoder = Decoder(layer_size, d_model, num_heads, num_triplet, use_ln=use_ln)

    def __call__(self, inputs, is_training):
        with tf.variable_scope(SCOPE):
            enc_i, enc_v, enc_t, enc_ivt = self.encoder(inputs=inputs, is_training=is_training)
            dec_ivt = self.decoder(enc_i, enc_v, enc_t, enc_ivt, is_training=is_training)
        return enc_i, enc_v, enc_t, dec_ivt


class Modules():
    def __init__(self):
        super(Modules, self).__init__()

    def conv2d(self, inputs, kernel_size, filters, name, strides=1, batch_norm=False, activation=True, rate=None, is_training=True):
        x_shape = inputs.get_shape().as_list()
        x = inputs
        with tf.variable_scope(name) as scope:
            w   = tf.get_variable(name='weights', shape=[kernel_size, kernel_size, int(x_shape[3]), filters])
            if rate == None:
                x = tf.nn.conv2d(input=x, filter=w,  padding='SAME', strides=[1, strides, strides, 1], name='conv')
            else:
                x = tf.nn.atrousconv2d(value=x,  filters=w, padding='SAME', rate=rate, name='conv')
            if batch_norm:
                with tf.variable_scope('BatchNorm'):
                    x = self.batch_norm(x, is_training=is_training)
            else:
                b = tf.get_variable(name='biases', shape=[filters])
                x = x + b
            if activation:
                x = self.elu(x)
            outputs = x
        print('\tBuilding unit: {}: {} --> {}'.format( scope.name, inputs.get_shape(),  outputs.get_shape()))
        return outputs

    def batch_norm(self, x, global_step=None, is_training=True, name='bn'):
        moving_average_decay = 0.9
        with tf.variable_scope(name):
            decay = moving_average_decay
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            with tf.device('/CPU:0'):
                mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32, initializer=tf.zeros_initializer(), trainable=False)
                sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32, initializer=tf.ones_initializer(), trainable=False)
                beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32, initializer=tf.zeros_initializer())
                gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32, initializer=tf.ones_initializer())
            # These ops will only be preformed when training.
            update = 1.0 - decay
            update_mu = mu.assign_sub(update*(mu - batch_mean))
            update_sigma = sigma.assign_sub(update*(sigma - batch_var))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)    
            mean, var = tf.cond(is_training, lambda: (batch_mean, batch_var), lambda: (mu, sigma))
            bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
        return bn

    def batch_norm_1d(self, x, global_step=None, is_training=True, name='bn'):
        moving_average_decay = 0.9
        with tf.variable_scope(name):
            decay = moving_average_decay
            batch_mean, batch_var = tf.nn.moments(x, [0, 1])
            with tf.device('/CPU:0'):
                mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32, initializer=tf.zeros_initializer(), trainable=False)
                sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32, initializer=tf.ones_initializer(), trainable=False)
                beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32, initializer=tf.zeros_initializer())
                gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32, initializer=tf.ones_initializer())
            # These ops will only be preformed when training.
            update = 1.0 - decay
            update_mu = mu.assign_sub(update*(mu - batch_mean))
            update_sigma = sigma.assign_sub(update*(sigma - batch_var))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)    
            mean, var = tf.cond(is_training, lambda: (batch_mean, batch_var), lambda: (mu, sigma))
            bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
        return bn    

    def elu(self, x, name='elu'):
        return tf.nn.elu(x, name=name)    

    def relu(self, x, name='relu'):
        return tf.nn.relu(x, name=name)        
    
    def sigmoid(self, x, name="sigmoid"):
        return tf.sigmoid(x, name=name)                

    def softmax(self, x, axis=-1, name="softmax"):
        return tf.nn.softmax(x, axis=axis, name=name)    

    def tanh(self, x, name='tanh'):
        return tf.nn.tanh(x, name=name)    
       
    def avg_pool(self, x, k=2, s=1, padding='SAME', name='avg_pool'):
        with tf.name_scope(name):
            return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding, name=name)

    def max_pool(self, x, k=4, s=1, padding='SAME', name='max_pool'):
        with tf.name_scope(name):
            return tf.nn.max_pool(value=x, ksize=[1,k,k,1], strides=[1,s,s,1], padding=padding, name=name)

    def wildcat_pool(self, x, k=4, s=1, padding='SAME'):
        with tf.name_scope(name):
            return tf.reduce_max(x, axis=[1,2]) + 0.6*tf.reduce_min(x, axis=[1,2], name='wildcat_pool')
             
    def global_avg_pool(self, x, k=4, s=1, padding='SAME', name='global_avg_pool'):
        with tf.name_scope(name):
            x = self.avg_pool(x, k=k, s=s, padding=padding, name=name)
            return tf.reduce_mean(x, axis=[1, 2])     
    
    def global_max_pool(self, x, k=4, s=1, padding='SAME', name='global_max_pool'):   
        with tf.name_scope(name): 
            x = self.max_pool(x, k=k, s=s, padding=padding, name=name)
            return tf.reduce_max(x, axis=[1,2])        

    def global_wildcat_pool(self, x, k=4, s=1, padding='SAME', name='global_wildcat_pool'):
        with tf.name_scope(name):
            return tf.reduce_max(x, axis=[1,2]) + 0.6*tf.reduce_min(x, axis=[1,2], name=name)    

    def flatten(self, inputs):
        shape   = inputs.get_shape().as_list()
        dim     = np.prod(shape[1:])         #dim     = tf.reduce_prod(tf.shape(inputs)[1:])
        return tf.reshape(inputs, shape=[-1, dim])

    def fc(self, inputs, units, name='dense'):
        with tf.variable_scope(name) as scope:
            x  = self.flatten(inputs)
            n  = x.get_shape().as_list()[-1]
            w  = tf.get_variable('weight', [n, units], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b  = tf.get_variable('bias', [units], initializer=tf.constant_initializer(0))
            outputs  = tf.matmul(x, w, name='fc') + b
            print('\tBuilding unit: {}: {} --> {}'.format( scope.name, inputs.get_shape(),  outputs.get_shape()))
        return outputs
        
    def dense(self, inputs, units, name="fc"):
        with tf.variable_scope(name) as scope:
            in_dims   = inputs.get_shape().as_list()[1:]
            in_units  = np.prod(in_dims)
            out_units = (units * in_units)/in_dims[-1]
            out_dims  = in_dims.copy()
            out_dims  = [-1]+out_dims
            out_dims[-1] = units
            weight    = tf.get_variable('weight', [in_units, out_units], initializer=tf.truncated_normal_initializer(stddev=0.02))
            bias      = tf.get_variable('bias', [out_units], initializer=tf.constant_initializer(0))
            inputs    = tf.reshape(inputs, shape=[-1, np.prod(inputs.get_shape().as_list()[1:]) ])
            outputs   = tf.matmul(inputs, weight, name='fc') + bias
            outputs   = tf.reshape(outputs, shape=out_dims)                
            print('\tBuilding unit: {}: {} --> {}'.format( scope.name, inputs.get_shape(),  outputs.get_shape()))
        return outputs


class Encoder(Modules):
    def __init__(self, basename, num_tool, num_verb, num_target, num_triplet, hr_output):
        super(Encoder, self).__init__()
        self.num_tool      = num_tool
        self.num_verb      = num_verb
        self.num_target    = num_target
        self.num_triplet   = num_triplet
        self.hr_output     = hr_output
        self.wsl           = self.tool_detector
        self.bottleneck    = self.bottleneck
        self.cagam         = self.cagam
        if basename        == 'resnet18':
            self.basemodel =  self.resnet18
        elif basename      == 'resnet50':
            self.basemodel = self.resnet50

    def __call__(self, inputs, is_training):
        with tf.variable_scope('encoder'):  
            high_x, low_x = self.basemodel(inputs, is_training, hr_output=self.hr_output)
            enc_i         = self.wsl(high_x, is_training)
            enc_v, enc_t  = self.cagam(high_x, enc_i[0], is_training)
            enc_ivt       = self.bottleneck(low_x, is_training)
        return enc_i, enc_v, enc_t, enc_ivt

    def resnet18(self, inputs, is_training=tf.constant(True), hr_output=False):  
        import resnet         
        with tf.variable_scope('backbone') as scope:
            resnet_network       = resnet.ResNet(images=inputs, version=18, is_train=is_training, hr_output=hr_output)
            high_features, end_points = resnet_network._build_model()
            low_features = end_points['block_2']
        print('\tBuilding unit: {}: {} --> {}'.format( scope.name, inputs.get_shape(),  high_features.get_shape()))
        return high_features, low_features
    
    def resnet50(self, inputs, is_training=tf.constant(True), hr_output=False): 
        from tensorflow.contrib.slim.nets import resnet_v2
        slim = tf.contrib.slim
        weight_decay       = 1e-5
        batch_norm_decay   = 0.9997
        batch_norm_epsilon = 1e-4
        with tf.variable_scope('backbone') as scope:
            with slim.arg_scope(
                 resnet_v2.resnet_arg_scope(
                         weight_decay=weight_decay,
                         batch_norm_decay=batch_norm_decay,
                         batch_norm_epsilon=batch_norm_epsilon )
                 ):
                 resnet = getattr(resnet_v2, 'resnet_v2_50')
                 _, end_points = resnet(inputs=inputs,
                                   is_training=is_training,
                                   num_classes=None,
                                   global_pool=False,
                                   output_stride=16,
                                   reuse=False)
                 high_features = end_points['{}/backbone/resnet_v2_50/block4'.format(SCOPE)]
                 low_features  = end_points['{}/backbone/resnet_v2_50/block1'.format(SCOPE)]
        print('\tBuilding unit: {}: {} --> {}'.format( scope.name, inputs.get_shape(),  high_features.get_shape()))
        return high_features, low_features

    def bottleneck(self, features, is_training):
        with tf.variable_scope('bottleneck'):  
            features = self.conv2d(features, kernel_size=3, filters=256, strides=2, name='conv1', batch_norm=True, activation=True, is_training=is_training)
            features = self.conv2d(features, kernel_size=1, filters=self.num_triplet, name='conv2'.format(self.num_triplet), batch_norm=True, activation=True, is_training=is_training)
        return features

    def tool_detector(self, features, is_training):
        with tf.variable_scope('wsl'):
            features    = self.conv2d(features, kernel_size=3, filters=64, name='conv1', batch_norm=True, activation=True, is_training=is_training)
            tool_maps   = self.conv2d(features, kernel_size=1, filters=self.num_tool, name='cam', batch_norm=False, activation=False, is_training=is_training)
            tool_logits = self.global_max_pool(tool_maps, name='gmp')
        return tool_maps, tool_logits

    def cagam(self, base_features, cam_features, is_training):
        with tf.variable_scope('cagam'):
            cam_channel_affinity       = self.get_channel_affinity_map(cam_features, name="cam", is_training=is_training)
            cam_position_affinity      = self.get_position_affinity_map(cam_features, name="cam", is_training=is_training)
            verb_maps, verb_logits     = self.cag_channel_attention(base_features, cam_channel_affinity, self.num_verb, is_training, name='verb_detection')
            target_maps, target_logits = self.cag_position_attention(base_features, cam_position_affinity, self.num_target, is_training, name='target_detection')  
        return (verb_maps, verb_logits), (target_maps, target_logits)

    def get_channel_affinity_map(self, x, name, is_training):
        z         = tf.identity(x)
        x_shape   = tf.shape(x)
        with tf.variable_scope(name+"_channel_affinity"):                
            query = self.conv2d(x, kernel_size=1, filters=x.shape[3], name='conv1x1_query', batch_norm=True, activation=True, is_training=is_training) #[B, H, W, C] QUERY
            key   = self.conv2d(z, kernel_size=1, filters=x.shape[3], name='conv1x1_key',   batch_norm=True, activation=True, is_training=is_training) #[B, H, W, C]  KEY                      
            query = tf.reshape(query, shape=[x_shape[0], x_shape[1]*x_shape[2], x_shape[3]]) #[B,N,C] where N = HxW
            key   = tf.reshape(key,   shape=[x_shape[0], x_shape[1]*x_shape[2], x_shape[3]]) #[B,N,C]
            alignment_score = tf.matmul(tf.transpose(query, [0,2,1]), key, name='channel_affinity') #[B,C,C]
        return alignment_score    
    
    def get_position_affinity_map(self, x, name, is_training):
        z         = tf.identity(x)
        x_shape   = tf.shape(x)
        with tf.variable_scope(name+"_position_affinity"):                
            query = self.conv2d(x, kernel_size=1, filters=x.shape[3], name='conv1x1_query', batch_norm=True, activation=True, is_training=is_training) #[B, H, W, C] QUERY
            key   = self.conv2d(z, kernel_size=1, filters=x.shape[3], name='conv1x1_key',   batch_norm=True, activation=True, is_training=is_training) #[B, H, W, C]  KEY                      
            query = tf.reshape(query, shape=[x_shape[0], x_shape[1]*x_shape[2], x_shape[3]]) #[B,N,C] where N = HxW
            key   = tf.reshape(key,   shape=[x_shape[0], x_shape[1]*x_shape[2], x_shape[3]]) #[B,N,C]
            alignment_score = tf.matmul(query, tf.transpose(key, [0,2,1]), name='position_affinity') #[B,N,N]
        return alignment_score

    def cag_position_attention(self, base_features, transfer_affinity, num_class, is_training, depth=64, name='target_subnet'):
        # A modified saliency guided self-attention network | https://arxiv.org/pdf/1910.05475v2.pdf
        # inputs: 
            # base_features: raw inputs
            # transfer_affinity:  heatmap for the guide [B,N,C]
        # output: attention logits
        with tf.variable_scope('{}'.format(name)):     
            spatial_features   = self.conv2d(base_features, kernel_size=3, filters=depth, name='conv_3x3x{}'.format(depth), batch_norm=True, activation=True, is_training=is_training)  
            receiver_affinity  = self.get_position_affinity_map(spatial_features, name="receiver", is_training=is_training) #[B,N,N]
            x_shape            = tf.shape(spatial_features)
            dk                 = tf.cast(x_shape[-1], tf.float32) 
            with tf.variable_scope('cag_position_attention'):
                beta        = tf.get_variable(name='beta', shape=(), trainable=True, initializer=tf.truncated_normal_initializer(stddev=0.02)) # temperature
                attention   = (receiver_affinity * transfer_affinity) / tf.sqrt(dk) 
                attention   = tf.nn.softmax(attention) 
                enhanced    = tf.matmul(tf.transpose(attention,[0,2,1]), tf.reshape(spatial_features, shape=[x_shape[0], x_shape[1]*x_shape[2], x_shape[3]])) * beta
                enhanced    = tf.reshape(enhanced, shape=[x_shape[0], x_shape[1], x_shape[2], x_shape[3]]) # [B,H,W,C]
                enhanced    = self.batch_norm(enhanced + spatial_features, is_training=is_training, name="bn_attention")      
                cmaps       = self.conv2d(enhanced, kernel_size=1, filters=num_class, name='conv_1x1x{}'.format(num_class), batch_norm=False, activation=False, is_training=is_training)        
                logits      = self.global_max_pool(cmaps)
            tf.summary.scalar('steps/attention_temperature', tf.identity(beta))
            # self._log_transfer_attention(enhanced_affinity)
        return cmaps, logits

    def cag_channel_attention(self, base_features, transfer_affinity, num_class, is_training, depth=6, name='verb_subnet'):
        # A modified saliency guided self-attention network | https://arxiv.org/pdf/1910.05475v2.pdf
        # inputs: 
            # base_features: raw inputs
            # transfer_affinity:  heatmap for the guide
        # output: attention logits
        with tf.variable_scope('{}'.format(name)):     
            spatial_features   = self.conv2d(base_features, kernel_size=3, filters=depth, name='conv_3x3x{}'.format(depth), batch_norm=True, activation=True, is_training=is_training)  
            receiver_affinity  = self.get_channel_affinity_map(spatial_features, name="receiver", is_training=is_training)
            x_shape            = tf.shape(spatial_features)
            dk                 = tf.cast(x_shape[-1], tf.float32) 
            with tf.variable_scope('cag_channel_attention'):  
                beta        = tf.get_variable(name='beta', shape=(), trainable=True, initializer=tf.truncated_normal_initializer(stddev=0.02)) # temperature
                attention   = (receiver_affinity * transfer_affinity)/tf.sqrt(dk)      
                attention   = tf.nn.softmax(attention)        
                enhanced    = tf.matmul(tf.reshape(spatial_features, shape=[x_shape[0], x_shape[1]*x_shape[2], x_shape[3]]), tf.transpose(attention,[0,2,1])) * beta
                enhanced    = tf.reshape(enhanced, shape=[x_shape[0], x_shape[1], x_shape[2], x_shape[3]]) # [B,H,W,C]
                enhanced    = self.batch_norm(enhanced + spatial_features, is_training=is_training, name="bn_attention")         
                cmaps       = self.conv2d(enhanced, kernel_size=1, filters=num_class, name='conv_1x1x{}'.format(num_class), batch_norm=False, activation=False, is_training=is_training)         
                logits      = self.global_max_pool(cmaps)
            tf.summary.scalar('steps/attention_temperature', tf.identity(beta))
        return cmaps, logits


class Decoder(Modules):
    def __init__(self, layer_size, d_model, num_heads, num_class=100, use_ln=False):
        super(Decoder, self).__init__()
        self.num_class  = num_class
        self.layer_size = layer_size
        self.num_heads  = num_heads
        self.d_model    = d_model
        self.use_ln      = use_ln
        self.mhma       = self.multihead_of_mixed_attention
        self.ffn        = self.feed_forward_network
        self.classifier = self.classification_layer

    def __call__(self, enc_i, enc_v, enc_t, enc_ivt, is_training):
        with tf.variable_scope('decoder'):   
            feat_i   = self.elu(self.batch_norm(enc_i[0], is_training=is_training, name='bn_I'), name='elu_I')
            feat_v   = self.elu(self.batch_norm(enc_v[0], is_training=is_training, name='bn_V'), name='elu_V')
            feat_t   = self.elu(self.batch_norm(enc_t[0], is_training=is_training, name='bn_T'), name='elu_T') 
            feat_ivt = tf.identity(enc_ivt)
            for layer in range(self.layer_size):
                k    = self.layer_size-layer-1
                with tf.variable_scope('layer_{}'.format(layer)):
                    x        = self.projection(feat_ivt, feat_i, feat_v, feat_t, depth=self.d_model, is_training=is_training)
                    feat_ivt = self.mhma(x, self.d_model, self.num_heads, num_class=self.num_class, is_training=is_training, use_ln=self.use_ln)
                    feat_ivt = self.ffn( feat_ivt, self.num_class, k=k, is_training=is_training, use_ln=self.use_ln)
            logits = self.classifier(feat_ivt, self.num_class)
        return logits

    def scale_dot_product(self, key, value, query):
        with tf.variable_scope("scale_dot_product"):
            dk     = tf.cast(tf.shape(key)[-1], tf.float32)
            attn_w = tf.matmul(key, query, transpose_b=True, name='affinity')                 
            attn_w = attn_w / tf.sqrt(dk)              
            attn_w = tf.nn.softmax(attn_w, name='activation')
            attn   = tf.matmul(value, attn_w, name='attention') 
        return attn

    def projection(self, feat_ivt, feat_i, feat_v, feat_t, depth, is_training):
        with tf.variable_scope("projection"):
            mean_ivt    = tf.reduce_mean(feat_ivt, axis=[1, 2], name='mean_ivt') 
            query       = tf.contrib.layers.fully_connected(mean_ivt, num_outputs=depth, activation_fn=tf.nn.elu, scope='query') # [B,D]
            query       = tf.contrib.layers.dropout(query, keep_prob=0.7, is_training=is_training, scope='query_dropout')
            key_ivt     = tf.contrib.layers.fully_connected(mean_ivt, num_outputs=depth, activation_fn=tf.nn.elu, scope='key_ivt') # [B,D]
            value_ivt   = self.conv2d(feat_ivt, kernel_size=1, filters=depth, name='value_ivt', batch_norm=True, activation=True, is_training=is_training) # [B,h,w,D]

            mean_i      = tf.reduce_mean(feat_i, axis=[1, 2], name='mean_i') 
            key_i       = tf.contrib.layers.fully_connected(mean_i, num_outputs=depth, activation_fn=tf.nn.elu, scope='key_i')
            value_i     = self.conv2d(feat_i, kernel_size=1, filters=depth, name='value_i', batch_norm=True, activation=True, is_training=is_training) # [B,h,w,D]

            mean_v      = tf.reduce_mean(feat_v, axis=[1, 2], name='mean_v') 
            key_v       = tf.contrib.layers.fully_connected(mean_v, num_outputs=depth, activation_fn=tf.nn.elu, scope='key_v')
            value_v     = self.conv2d(feat_v, kernel_size=1, filters=depth, name='value_v', batch_norm=True, activation=True, is_training=is_training) # [B,h,w,D]

            mean_t      = tf.reduce_mean(feat_t, axis=[1, 2], name='mean_t') 
            key_t       = tf.contrib.layers.fully_connected(mean_t, num_outputs=depth, activation_fn=tf.nn.elu, scope='key_t')
            value_t     = self.conv2d(feat_t, kernel_size=1, filters=depth, name='value_t', batch_norm=True, activation=True, is_training=is_training) # [B,h,w,D]

            shapes      = value_i.get_shape().as_list()
            value_ivt   = self.elu(tf.image.resize_bilinear(value_ivt, size=[shapes[1], shapes[2]]))
            feat_ivt    = self.elu(tf.image.resize_bilinear(feat_ivt,  size=[shapes[1], shapes[2]]))
        return (feat_ivt, (key_i,value_i), (key_v,value_v), (key_t,value_t), (query,key_ivt,value_ivt))

    def multihead_of_mixed_attention(self, x, depth, head, num_class, is_training, use_ln=False):
        (feat_ivt, (key_i,value_i), (key_v,value_v), (key_t,value_t), (query,key_ivt,value_ivt)) = x
        b,h,w,c   = feat_ivt.get_shape().as_list()
        # symbols (b = batch, h= height, w=width, c=channel or class, H=heads, D=depth)
        with tf.variable_scope("mhma"):
            z     = tf.identity(feat_ivt)
            # query
            query = tf.stack([query]*head, axis=1, name='query') # [b,H,D]
            # query = tf.contrib.layers.dropout(query, keep_prob=0.7, is_training=is_training, scope='query_dropout')
            query = tf.reshape(query, (-1,head,depth,1))  # [b,H,D,1]
            # key
            key   = tf.stack([key_ivt, key_i, key_v, key_t], axis=1) # [b,H,D]
            key   = tf.reshape(key, (-1,head,depth,1))  # [b,H,D,1]
            # value
            value = tf.stack([value_ivt, value_i, value_v, value_t], axis=1)  #[b,H,h,w,D]
            value = tf.reshape(value, (-1,head,h*w,depth))  # [b,H,hw,D]   
            # attention
            attn  = self.scale_dot_product(key, value, query)  # [b,H,hw,D]
            attn  = tf.transpose(attn, perm=[0, 2, 1, 3]) # [b,hw,D,H]
            attn  = tf.reshape(attn, shape=[-1,h*w,depth*head])  # [b,hw,DH]
            attn  = tf.reshape(attn, shape=[-1,h,w,depth*head])  # [b,h,w,DH]
            # addnorm
            mha   = self.conv2d(attn, kernel_size=3, filters=num_class, name='conv_1x1x{}'.format(num_class), batch_norm=True, activation=True, is_training=is_training) # [b,h,w,c]
            mha   = tf.contrib.layers.layer_norm((mha + z), activation_fn=tf.nn.elu, scope="layer_norm") if use_ln else self.batch_norm((mha + z), is_training=is_training, name="layer_norm")
        return mha

    def feed_forward_network(self, x, depth, k, is_training, use_ln=False):
        act = True if k>0 else False  
        with tf.variable_scope("feed_forward"):
            z = tf.identity(x)
            y = self.conv2d(x, kernel_size=3, filters=depth, name='conv3x3x{}a'.format(depth), batch_norm=True, activation=True, is_training=is_training)
            y = self.conv2d(y, kernel_size=1, filters=depth, name='conv1x1x{}b'.format(depth), batch_norm=True, activation=act, is_training=is_training)
            y = tf.contrib.layers.layer_norm((y + z), activation_fn=tf.nn.elu, scope="layer_norm") if use_ln else self.batch_norm((y + z), is_training=is_training, name="layer_norm") 
        return y

    def classification_layer(self, inputs, num_class):
        with tf.variable_scope('classifier'):
            pooled  = self.global_max_pool(inputs, name='gmp')
            logits  = self.fc(pooled, units=num_class, name='mlp')                  
        return logits