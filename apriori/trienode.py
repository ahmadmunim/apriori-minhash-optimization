class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False  # Marks the end of an itemset

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, itemset):
        node = self.root
        for item in sorted(itemset):  # Sorting ensures consistent paths
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]
        node.is_end = True

    def search(self, itemset):
        node = self.root
        for item in sorted(itemset):
            if item not in node.children:
                return False
            node = node.children[item]
        return node.is_end
