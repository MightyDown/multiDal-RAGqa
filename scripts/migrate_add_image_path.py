"""给所有现有 Milvus collection 添加 image_path 字段。"""
from pymilvus import Collection, connections, utility

connections.connect(host="milvus", port="19530")

collections = utility.list_collections()
print(f"Found {len(collections)} collections: {collections}")

for name in collections:
    try:
        coll = Collection(name)
        coll.load()
        coll.alter_field("image_path", "data_type", "VarChar", {"max_length": 512})
        print(f"  ALTERED {name} (after load)")
    except Exception as e:
        if "image_path" in str(e):
            print(f"  SKIP {name}: image_path may already exist or other error: {e}")
        else:
            print(f"  ERROR {name}: {e}")

connections.disconnect("default")
print("Done")