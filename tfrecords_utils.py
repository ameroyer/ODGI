import tensorflow as tf


def _int64_feature(value):
    """TFRecords int feature"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """TFRecords float feature"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_feature_write(key, value):
    """Choose the right feature function for the given key to write to TFRecords
    
    Args:
        key: the feature name
        value: the feature value
    """
    if key in ['im_id', 'num_boxes']:
        return _int64_feature([value])
    elif key in['bounding_boxes']:
        return _float_feature(value.flatten())
    elif key in ['classes']:
        return _int64_feature(value.flatten())
    else:
        raise SystemExit("Unknown feature %s" % key)    
    
    
def write_tfrecords(features_list):
    """Returns a dictionnary of tf.Feature to write in an TFRecords Example
    
    Args:
        feature_list: A list of (key string, feature value) pairs
    """
    return {key: get_feature_write(key, value) for key, value in features_list if value is not None}

    
def get_feature_read(key, max_num_bbs=None):
    """Choose the right feature function for the given key to parse TFRecords
    
    Args:
        key: the feature name
        max_num_bbs: Max number of bounding boxes (used for `bounding_boxes` and `classes`)
        max_num_groups: Number of pre-defined groups (used for `clustered_bounding_boxes`)
    """
    if key in ['im_id', 'num_boxes']:
        return tf.FixedLenFeature((), tf.int64)
    elif key in ['bounding_boxes']:
        assert max_num_bbs is not None
        return tf.FixedLenFeature((max_num_bbs, 4), tf.float32)
    elif key in ['classes']:
        assert max_num_bbs is not None
        return tf.FixedLenFeature((max_num_bbs,), tf.int64) 
    else:
        raise SystemExit("Unknown feature", key)    
    
    
def read_tfrecords(keys_list, max_num_bbs=None):
    """Create a TFRecords feature from a list of pair (key, value). Same Kwargs as get_feature_read
    
    Args:
        feature_list: A list of key strings corresponding to entries in the records
    """
    return {key: get_feature_read(key, max_num_bbs=max_num_bbs) for key in keys_list}