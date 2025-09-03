import tensorflow as tf

path = "/home/freddie/project/openvla-oft/modified_libero_rlds/libero_spatial_no_noops/1.0.0/libero_spatial-train.tfrecord-00000-of-00016"  # update if needed

def try_example(rec_bytes):
    ex = tf.train.Example()
    ex.ParseFromString(rec_bytes)
    keys = sorted(ex.features.feature.keys())
    return "Example", keys, ex

def try_seq_example(rec_bytes):
    ex = tf.train.SequenceExample()
    ex.ParseFromString(rec_bytes)
    # sequence features are nested; print context keys first
    ctx_keys = sorted(ex.context.feature.keys())
    feat_list_keys = sorted(ex.feature_lists.feature_list.keys())
    return "SequenceExample", (ctx_keys, feat_list_keys), ex

raw = next(iter(tf.data.TFRecordDataset([path])))
rec_bytes = raw.numpy()

kind, keys, ex = try_example(rec_bytes)
if kind == "Example" and keys:
    print("[Parsed as tf.train.Example]")
    for k in keys:
        if any(s in k.lower() for s in ["lang", "instr", "task", "desc"]):
            print("  ", k)
    # Try to print a few likely fields as strings
    def get_bytes(key):
        v = ex.features.feature[key].bytes_list.value
        return [b.decode("utf-8", errors="ignore") for b in v]
    for k in keys:
        if "language" in k.lower() or "instruction" in k.lower() or "task" in k.lower() or "description" in k.lower():
            try:
                print(f"\nSample for {k}:")
                print(get_bytes(k)[:3])
            except Exception:
                pass
else:
    kind, (ctx_keys, fl_keys), ex = try_seq_example(rec_bytes)
    print("[Parsed as tf.train.SequenceExample]")
    print("Context keys:")
    for k in ctx_keys:
        if any(s in k.lower() for s in ["lang", "instr", "task", "desc"]):
            print("  ", k)
    print("Feature list keys:")
    for k in fl_keys:
        if any(s in k.lower() for s in ["lang", "instr", "task", "desc"]):
            print("  ", k)
