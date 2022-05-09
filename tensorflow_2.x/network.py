#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 19:09:36 2021

@author: nwoye
"""

import tensorflow as tf

SCOPE           = 'rendezvous'
INPUT_SHAPE     = (256,448,3)
OUTPUT_SHAPE    = (256,448,3)
NETSCOPE        = {
            'mobilenet':{
                    'high_level_feature':'out_relu', 
                    'low_level_feature':'block_1_project_BN', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'mobilenetv2':{
                    'high_level_feature':'out_relu', 
                    'low_level_feature':'block_1_project_BN', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'xception':{
                    'high_level_feature':'block14_sepconv2_act', 
                    'low_level_feature':'block1_conv2_act', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'resnet50':{
                    'high_level_feature':'conv5_block3_out', 
                    'low_level_feature':'pool1_pool', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'resnet50v2':{
                    'high_level_feature':'post_relu', 
                    'low_level_feature':'pool1_pool', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'resnet18v2':{
                    'high_level_feature':'post_relu', 
                    'low_level_feature':'pool1_pool', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)
                    },
            'densenet169':{
                    'high_level_feature':'bn', 
                    'low_level_feature':'pool1', 
                    'low_output_shape_ratio':(4,4), 
                    'high_output_shape_ratio':(8,8)

                    }
        }    


class Rendezvous(tf.keras.Model):
    """
    Rendezvous: attention mechanism for surgical action triplet recognition by Nwoye, C.I. et.al. 2021
    @args:
        image_shape: a tuple (height, width) e.g: (224,224)
        basename: Feature extraction network: e.g: "resnet50", "VGG19"
        num_tool: default = 6, 
        num_verb: default = 10, 
        num_target: default = 15, 
        num_triplet: default = 100, 
        layer_size: multi-head layers default = 8, 
        num_heads: default = 4, 
        d_model: mh-SCAM's feature depth. default = 128
    @call:
        inputs: Batch of input images of shape [batch, height, width, channel]
        training: True or False. Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (doing nothing)
    @output: 
        enc_i: tuple (cam, logits) for instrument
        enc_v: tuple (cam, logits) for verb
        enc_t: tuple (cam, logits) for target
        dec_ivt: logits for triplet
    """
    def __init__(self, image_shape=(256,448,3), basename="resnet50", pretrained='imagenet', num_tool=6, num_verb=10, num_target=15, num_triplet=100, layer_size=8, num_heads=4, d_model=128, use_ln=False):
        super(Rendezvous, self).__init__()
        inputs          = tf.keras.Input(shape=image_shape)
        self.encoder    = Encoder(basename, pretrained, image_shape, num_tool, num_verb, num_target, num_triplet)
        self.decoder    = Decoder(layer_size, d_model, num_heads, num_triplet, use_ln)
        enc_i, enc_v, enc_t, enc_ivt = self.encoder(inputs)
        dec_ivt         = self.decoder(enc_i, enc_v, enc_t, enc_ivt)
        self.rendezvous = tf.keras.models.Model(inputs=inputs, outputs=(enc_i, enc_v, enc_t, dec_ivt), name='rendezvous')

    def call(self, inputs, training):
        enc_i, enc_v, enc_t, dec_ivt = self.rendezvous(inputs, training=training)
        return enc_i, enc_v, enc_t, dec_ivt
      

# Model Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, basename, pretrained, image_shape=(256,448,3), num_tool=6, num_verb=10, num_target=15, num_triplet=100):
        super(Encoder, self).__init__()
        self.basemodel  = Basemodel(basename, image_shape, pretrained)
        self.wsl        = WSL(num_tool)
        self.cagam      = CAGAM(num_tool, num_verb, num_target)
        self.bottleneck = Bottleneck(num_triplet)

    def call(self, inputs, training):
        low_x, high_x   = self.basemodel(inputs, training)
        enc_i           = self.wsl(high_x, training)
        enc_v, enc_t    = self.cagam(high_x, enc_i[0], training)
        enc_ivt         = self.bottleneck(low_x, training)
        return enc_i, enc_v, enc_t, enc_ivt


# Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, layer_size=8, d_model=128, num_heads=4, num_class=100, use_ln=False):
        super(Decoder, self).__init__()
        self.projection = [Projection(d_model) for i in range(layer_size)]
        self.mhma       = [MHMA(num_class, num_heads, use_ln) for i in range(layer_size)]
        self.ffnet      = [FFN(layer_size-i-1, num_class, use_ln) for i in range(layer_size)]
        self.classifier = Classifier(num_class)

    def call(self, enc_i, enc_v, enc_t, enc_ivt, training):
        X = tf.identity(enc_ivt)
        for P, M, F in zip(self.projection, self.mhma, self.ffnet):
            X = P(enc_i[0], enc_v[0], enc_t[0], X, training)
            X = M(X, training)
            X = F(X, training)
        logits_ivt = self.classifier(X, training)
        return logits_ivt


# Backbone
class Basemodel(tf.keras.layers.Layer):
    def __init__(self, basename, image_shape, pretrained='imagenet'):
        super(Basemodel, self).__init__()
        if basename ==  'mobilenet':
            base_model = tf.keras.applications.MobileNetV2(
                                    input_shape=image_shape,
                                    include_top=False,
                                    weights='imagenet')
        elif basename ==  'mobilenetv2':
            base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
                                    input_shape=image_shape,
                                    include_top=False,
                                    weights='imagenet')
        elif basename ==  'xception':
            base_model = tf.keras.applications.Xception(
                                    weights='imagenet',  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename ==  'resnet50':
            base_model = tf.keras.applications.resnet50.ResNet50(
                                    weights=pretrained,  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename ==  'resnet50v2':
            base_model = tf.keras.applications.resnet_v2.ResNet50V2(
                                    weights='imagenet',  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename ==  'resnet18v2':
            import resnet_v2
            base_model = resnet_v2.ResNet18V2(
                                    weights=None,  # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    stride=1,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        elif basename =='densenet169':
            base_model = tf.keras.applications.densenet.DenseNet169(
                                    include_top=False, 
                                    weights='imagenet',
                                    input_shape=image_shape )
        else: base_model = tf.keras.applications.resnet18.ResNet18( # not impl.
                                    weights='imagenet', # Load weights pre-trained on ImageNet.
                                    input_shape=image_shape,
                                    include_top=False)  # Do not include the ImageNet classifier at the top.
        self.base_model = tf.keras.models.Model(inputs=base_model.input, 
                                                outputs=(base_model.get_layer(NETSCOPE[basename]['low_level_feature']).output, base_model.output),
                                                name='backbone')
        # self.base_model.trainable = trainable        

    def call(self, inputs, training):
        return self.base_model(inputs, training=training)
            
            
# WSL of Tools
class WSL(tf.keras.layers.Layer):
    def __init__(self, num_class, depth=64):
        super(WSL, self).__init__()
        self.num_class = num_class
        self.conv1 = tf.keras.layers.Conv2D(depth, 3, activation=None, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(num_class, 1, activation=None, name='cam')
        self.gmp   = tf.keras.layers.GlobalMaxPooling2D()
        self.bn    = tf.keras.layers.BatchNormalization()
        self.elu   = tf.keras.activations.elu

    def call(self, inputs, training):
        feature = self.conv1(inputs, training=training)
        feature = self.elu(self.bn(feature, training=training))
        cam     = self.conv2(feature, training=training)
        logits  = self.gmp(cam)
        return cam, logits


# Class Activation Guided Attention Mechanism
class CAGAM(tf.keras.layers.Layer):
    def __init__(self, num_tool, num_verb, num_target):
        super(CAGAM, self).__init__()
        depth          = num_tool
        self.context1  = tf.keras.layers.Conv2D(depth, 3, activation=None, padding='same', name='verb/context')
        self.context2  = tf.keras.layers.Conv2D(depth, 3, activation=None, padding='same', name='target/context')
        self.q1        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='verb/query')
        self.q2        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='verb_tool/query')
        self.q3        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='target/query')
        self.q4        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='target_tool/query')
        self.k1        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='verb/key')
        self.k2        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='verb_tool/key')
        self.k3        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='target/key')
        self.k4        = tf.keras.layers.Conv2D(depth, 1, activation=None, name='target_tool/key')
        self.cmap1     = tf.keras.layers.Conv2D(num_verb, 1, activation=None, name='verb/cmap')
        self.cmap2     = tf.keras.layers.Conv2D(num_target, 1, activation=None, name='target/cmap')
        self.gmp       = tf.keras.layers.GlobalMaxPooling2D()
        self.soft      = tf.keras.layers.Softmax()
        self.elu       = tf.keras.activations.elu
        self.beta1     = self.add_weight("encoder/cagam/verb/beta", shape=())
        self.beta2     = self.add_weight("encoder/cagam/target/beta", shape=())       
        self.bn1       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_1')
        self.bn2       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_2')
        self.bn3       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_3')
        self.bn4       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_4')
        self.bn5       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_5')
        self.bn6       = tf.keras.layers.BatchNormalization(name='verb/batch_normalization_6')
        self.bn7       = tf.keras.layers.BatchNormalization(name='target/batch_normalization_1')
        self.bn8       = tf.keras.layers.BatchNormalization(name='target/batch_normalization_2')
        self.bn9       = tf.keras.layers.BatchNormalization(name='target/batch_normalization_3')
        self.bn10      = tf.keras.layers.BatchNormalization(name='target/batch_normalization_4')
        self.bn11      = tf.keras.layers.BatchNormalization(name='target/batch_normalization_5')
        self.bn12      = tf.keras.layers.BatchNormalization(name='target/batch_normalization_6')

    def get_verb(self, raw, cam, training):
        x = self.elu(self.bn1(self.context1(raw, training=training), training=training))
        z = tf.identity(x)               
        q1 = self.elu(self.bn2(self.q1(x, training=training), training=training))
        k1 = self.elu(self.bn3(self.k1(x, training=training), training=training))
        s  = k1.get_shape().as_list()
        w1 = tf.matmul(tf.reshape(q1,[-1,s[1]*s[2],s[3]]), tf.reshape(k1,[-1,s[1]*s[2],s[3]]), transpose_a=True)        
        q2 = self.elu(self.bn4(self.q2(cam, training=training), training=training))
        k2 = self.elu(self.bn5(self.k2(cam, training=training), training=training))
        s  = k2.get_shape().as_list()
        dk = tf.cast(s[-1], tf.float32)
        w2 = tf.matmul(tf.reshape(q2,[-1,s[1]*s[2],s[3]]), tf.reshape(k2,[-1,s[1]*s[2],s[3]]), transpose_a=True)
        attention = (w1 * w2) / tf.sqrt(dk)  
        attention = self.soft(attention) 
        s = z.get_shape().as_list()        
        v = tf.reshape(z, [-1, s[1]*s[2], s[3]])
        e = tf.matmul(v, attention) * self.beta1
        e = tf.reshape(e, [-1, s[1], s[2], s[3]]) 
        e = self.bn6(e + z)
        cmap = self.cmap1(e, training=training)
        y = self.gmp(cmap)
        return cmap, y

    def get_target(self, raw, cam, training):    
        x = self.elu(self.bn7(self.context2(raw, training=training), training=training))
        z = tf.identity(x)       
        q3 = self.elu(self.bn8(self.q3(x, training=training), training=training))
        k3 = self.elu(self.bn9(self.k3(x, training=training), training=training))
        s  = k3.get_shape().as_list()
        w3 = tf.matmul(tf.reshape(q3,[-1,s[1]*s[2],s[3]]), tf.reshape(k3,[-1,s[1]*s[2],s[3]]), transpose_b=True)  
        q4 = self.elu(self.bn10(self.q4(cam, training=training), training=training))
        k4 = self.elu(self.bn11(self.k4(cam, training=training), training=training))
        s  = k4.get_shape().as_list()
        dk = tf.cast(s[-1], tf.float32)
        w4 = tf.matmul(tf.reshape(q4,[-1,s[1]*s[2],s[3]]), tf.reshape(k4,[-1,s[1]*s[2],s[3]]), transpose_b=True)
        attention = (w3 * w4) / tf.sqrt(dk)  
        attention = self.soft(attention)
        s = z.get_shape().as_list() 
        v = tf.reshape(z, [-1, s[1]*s[2], s[3]])
        e = tf.matmul(attention, v) * self.beta2
        e = tf.reshape(e, [-1, s[1], s[2], s[3]])
        e = self.bn12(e + z)
        cmap = self.cmap2(e, training=training)
        y = self.gmp(cmap)
        return cmap, y

    def call(self, inputs, cam, training):
        cam_v, logit_v = self.get_verb(inputs, cam, training)
        cam_t, logit_t = self.get_target(inputs, cam, training)
        return (cam_v, logit_v), (cam_t, logit_t)


# Botleneck layer
class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, num_class, factor=1):
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(256, 3, strides=(2,2), activation=None, name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(num_class, 1, strides=(1,1), activation=None, name='conv2')
        self.elu   = tf.keras.activations.elu
        self.bn1   = tf.keras.layers.BatchNormalization()
        self.bn2   = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training):
        x = self.elu(self.bn1(self.conv1(inputs, training=training), training=training))
        x = self.elu(self.bn2(self.conv2(x, training=training), training=training))
        return x
    

# Projection function
class Projection(tf.keras.layers.Layer):
    def __init__(self, depth=128):
        super(Projection, self).__init__()   
        self.v    = tf.keras.layers.Conv2D(depth, 1, activation=None, name='ivt/value')     
        self.k    = tf.keras.layers.Dense(depth, activation=None, name='ivt/key')
        self.q    = tf.keras.layers.Dense(depth, activation=None, name='ivt/query')
        self.gap  = tf.keras.layers.GlobalAveragePooling2D()
        self.drop = tf.keras.layers.Dropout(rate=0.3)
        self.v1   = tf.keras.layers.Conv2D(depth, 1, activation=None, name='i/value')     
        self.k1   = tf.keras.layers.Dense(depth, activation=None, name='i/key')
        self.v2   = tf.keras.layers.Conv2D(depth, 1, activation=None, name='v/value')     
        self.k2   = tf.keras.layers.Dense(depth, activation=None, name='v/key')
        self.v3   = tf.keras.layers.Conv2D(depth, 1, activation=None, name='t/value')     
        self.k3   = tf.keras.layers.Dense(depth, activation=None, name='t/key')
        self.bn1  = tf.keras.layers.BatchNormalization()
        self.bn2  = tf.keras.layers.BatchNormalization()
        self.bn3  = tf.keras.layers.BatchNormalization()
        self.bn4  = tf.keras.layers.BatchNormalization()
        self.bn5  = tf.keras.layers.BatchNormalization()
        self.bn6  = tf.keras.layers.BatchNormalization()
        self.bn7  = tf.keras.layers.BatchNormalization()
        self.bn8  = tf.keras.layers.BatchNormalization()
        self.bn9  = tf.keras.layers.BatchNormalization()
        self.elu  = tf.keras.activations.elu
        
    def call(self, cam_i, cam_v, cam_t, X, training):
        q  = self.elu(self.bn1(self.q(self.drop(self.gap(X)), training=training), training=training)  )
        k  = self.elu(self.bn2(self.k(self.gap(X), training=training), training=training)) 
        v  = self.elu(self.bn3(self.v(X, training=training), training=training) )
        k1 = self.elu(self.bn4(self.k1(self.gap(cam_i), training=training), training=training)) 
        v1 = self.elu(self.bn5(self.v1(cam_i, training=training), training=training) )
        k2 = self.elu(self.bn6(self.k2(self.gap(cam_v), training=training), training=training)) 
        v2 = self.elu(self.bn7(self.v2(cam_v, training=training), training=training) )
        k3 = self.elu(self.bn8(self.k3(self.gap(cam_t), training=training), training=training) )
        v3 = self.elu(self.bn9(self.v3(cam_t, training=training), training=training) )
        s  = v1.get_shape().as_list()
        v  = self.elu(tf.image.resize(v, (s[1],s[2])))
        X  = self.elu(tf.image.resize(X, (s[1],s[2])))
        return (X, (k1,v1), (k2,v2), (k3,v3), (q,k,v))


# Multi-head of self and cross attention
class MHMA(tf.keras.layers.Layer):
    def __init__(self, num_class, num_heads=4, use_ln=False):
        super(MHMA, self).__init__()
        self.heads = num_heads
        self.conv  = tf.keras.layers.Conv2D(num_class, 3, activation=None, padding='same', name='concat')  
        self.bn    = tf.keras.layers.BatchNormalization()
        self.ln    = tf.keras.layers.LayerNormalization() if use_ln else tf.keras.layers.BatchNormalization()
        self.soft  = tf.keras.layers.Softmax()
        self.elu   = tf.keras.activations.elu

    def scale_dot_product(self, key, value, query):
        dk     = tf.cast(tf.shape(key)[-2], tf.float32)
        attn_w = tf.matmul(key, query, transpose_b=True, name='affinity')                 
        attn_w = attn_w / tf.sqrt(dk)              
        attn_w = self.soft(attn_w)
        attn   = tf.matmul(value, attn_w, name='attention') 
        return attn

    def call(self, inputs, training):
        (X, (k1,v1), (k2,v2), (k3,v3), (q,k,v)) = inputs
        query = tf.stack([q]*self.heads, axis=1, name='query') # [B,Head,D]
        query = tf.expand_dims(query, axis=-1) # [B,Head,D,1]
        key   = tf.stack([k,k1,k2,k3], axis=1, name='key') # [B,Head,D]
        key   = tf.expand_dims(key, axis=-1) # [B,Head,D,1]
        value = tf.stack([v,v1,v2,v3], axis=1, name='value') # [B,Head,H,W,D]
        dims  = value.get_shape().as_list() # [B,Head,H,W,D]
        value = tf.reshape(value, shape=[-1,dims[1],dims[2]*dims[3],dims[4]])# [B,Head,HW,D]   
        attn  = self.scale_dot_product(key, value, query)  # [B,Head,HW,D]
        attn  = tf.transpose(attn, perm=[0, 2, 1, 3]) # [B,HW,D,Head]        
        attn  = tf.reshape(attn, shape=[-1,dims[2]*dims[3],dims[1]*dims[4]])  # [B,HW,DHead]
        attn  = tf.reshape(attn, shape=[-1,dims[2],dims[3],dims[1]*dims[4]])  # [B,H,W,DHead]    
        mhma  = self.elu(self.bn(self.conv(attn, training=training), training=training) )
        mhma  = self.ln(mhma + tf.identity(X))
        return mhma


# Feed-forward layer
class FFN(tf.keras.layers.Layer):
    def __init__(self, k, num_class=100, use_ln=False):
        super(FFN, self).__init__()    
        def ignore(x):
            return x
        self.conv1 = tf.keras.layers.Conv2D(num_class, 3, activation=None, padding='same', name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(num_class, 1, activation=None, padding='same', name='conv2')
        self.elu1  = tf.keras.activations.elu
        self.elu2  = tf.keras.activations.elu if k>0 else ignore
        self.ln    = tf.keras.layers.LayerNormalization() if use_ln else tf.keras.layers.BatchNormalization()
        self.bn1   = tf.keras.layers.BatchNormalization()
        self.bn2   = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training):
        x  = self.elu1(self.bn1(self.conv1(inputs, training=training), training=training))
        x  = self.elu2(self.bn2(self.conv2(x, training=training), training=training))
        x  = self.ln(x + tf.identity(inputs))
        return x


# Classification layer
class Classifier(tf.keras.layers.Layer):
    def __init__(self, layer_size):
        super(Classifier, self).__init__()
        self.gmp  = tf.keras.layers.GlobalMaxPooling2D()
        self.fc   = tf.keras.layers.Dense(layer_size, name='mlp')

    def call(self, inputs, training):
        x = self.gmp(inputs)
        y = self.fc(x)
        return y