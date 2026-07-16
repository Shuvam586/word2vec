class HuffmanNode:
    def __init__(self, freq, data = None, left = None, right = None, node_id = None):
        self.freq = freq
        self.data = data
        self.left = left
        self.right = right
        self.node_id = node_id
    
    def __lt__(self, other):
        return self.freq < other.freq

    def __gt__(self, other):
        return self.freq > other.freq

def build_huffman_tree(word_counts: dict) -> HuffmanNode:
    nodes = []

    id = 1

    for word in word_counts:
        
        nodes.append(HuffmanNode(
            freq = word_counts[word],
            data = word,
            node_id = None
        ))
    
    while (len(nodes)>1):
        nodes.sort()
        node_a, node_b = nodes.pop(0), nodes.pop(0)
        parent = HuffmanNode(freq=node_a.freq+node_b.freq, left=node_a, right=node_b, node_id=id)
        nodes.append(parent)
        id += 1
    
    return nodes[0]

def generate_codes(root: HuffmanNode, word_to_id: dict) -> dict:
    queue = [["0", root.left], ["1", root.right]]

    d = {}

    while (len(queue)!=0):
        curr_item_path, curr_item = queue.pop(0)
        if (curr_item.data):
            d[word_to_id[curr_item.data]] = [int(i) for i in list(curr_item_path)]
            # print(curr_item.data, [int(i) for i in list(curr_item_path)])
        else:
            queue.extend([[curr_item_path+"0", curr_item.left], [curr_item_path+"1", curr_item.right]])

    # print(d)

    return d

def generate_paths(root:HuffmanNode, word_to_id: dict) -> dict:
    queue = [[[root.node_id], root.left], [[root.node_id], root.right]]

    d = {}

    while (len(queue)!=0):
        curr_item_path, curr_item = queue.pop(0)
        if (curr_item.data):
            d[word_to_id[curr_item.data]] = curr_item_path
            # print(curr_item.data, [int(i) for i in list(curr_item_path)])
        else:
            queue.extend([[curr_item_path+[curr_item.node_id], curr_item.left], [curr_item_path+[curr_item.node_id], curr_item.right]])

    # print(d)

    return d