# =========================================================================== #
class Tree :
    def __init__ (self, rootName) :
        self.rootName = rootName
        self.subtrees = []
    
    def __getitem__ (self, index) :
        # we don't need any checks for index value or type; this is allready handled by class list.
        return self.subtrees[index]
    
    def __setitem__ (self, index, value) :
        self.subtrees[index].rootname = value
    
    def __str__ (self) :
        return str(self.rootName)
    
    def __iter__ (self) :
        return TreeIterator(self)
    
    def checkIndex (self, index) :
        L = len(self.subtrees)
        
        if type(index) != int :
            raise TypeError("Index must be of type int!")
        
        if not (-(L + 1) <= index <= L) :
            raise IndexError("Index out of bounds!")
    
    def addNode (self, value, index = None) :
        # the optional argument index specifies where to add the new node.
        # it will be added *before* index, and allows negative indices, as with lists.
        # If set to none, the new item will be added to the back of the list.
        
        if index == None :
            index = len(self.subtrees)
        self.checkIndex(index)
        
        node = Tree(value)
        self.subtrees.insert(index, node)
    
    def removeNode(self, index) :
        self.checkIndex(index)
        
        if index < 0 :
            index += len(self.subtrees)
        
        self.subtrees = self.subtrees[:index] + self.subtrees[index + 1:]
        pass

# =========================================================================== #
class TreeIterator :
    def __init__ (self, tree) :
        self.tree     = tree                 # a reference to the tree we're going to traverse
        self.indices  = []                   # where are we at right now?
        self.limits   = []                   # how many nodes are on the other levels in the current subtree?
        self.explore  = True                 # ugly hack... see below...
        
    def resolveIndices (self) :
        currentNode = self.tree
        
        for index in self.indices :
            currentNode = currentNode[index]
        
        return currentNode
    
    def advanceIndices (self) :
        # first, try to go one level down:
        if self.explore :
            currentNode = self.resolveIndices()
            childCount  = len(currentNode.subtrees)
            if childCount :
                self.indices.append(0)
                self.limits .append(childCount)
                return
        else :
            self.explore = True
    
        # if no level down exists, increase the current counter
        self.indices[-1] += 1
        if self.indices[-1] == self.limits[-1] :           # we have reached the end of the current level. So...
            self.indices = self.indices[:-1]               # remove the end of our counter ...
            self.limits  = self.limits [:-1]               # and the current counter limit
            
            # have we removed the last level? Then we're done!
            if len(self.indices) == 0 :
                raise StopIteration
            
            self.explore = False                           # and make it so that we go one step ahead in the level one above
                                                           # it could be we ascend more than one level this way, hence the
                                                           # bizarre mechanic with this boolean.
            self.advanceIndices()
    
    def __next__ (self) :
        currentNode = self.resolveIndices()
        
        reVal = currentNode.rootName
        level = len(self.indices)
        
        self.advanceIndices()
        
        return level, reVal
    
    def __iter__ (self) :
        # Imagine, you want to output the tree structure, but omit the root.
        # Due to this dunder here, you can do so, simply by typing:
        
        # treeIter = iter(tree)
        # next(treeIter)                        # skip the first element
        # for indent, content in treeIter :     # for cals __iter__ of treeIter!
        #     ...
        
        return self

# =========================================================================== #
# Test environment

tree = Tree("files")



print("### ROOT ELEMENT:")
print(tree)
print()

print("### ADDING FIRST LEVEL CHILDREN...", end = "")
for node in ("Documents", "Pictures", "Downloads", "Music", "Misc") :
    tree.addNode(node)
print("DONE!")
print()

print("### ADDING SECOND LEVEL CHILDREN...", end = "")
nodeDocs = tree[0]
for node in ("Codes", "Ebooks", "Uni", "Bills and Money") :
    nodeDocs.addNode(node)

nodeMusic = tree[-2]
for node in ("no music at all", "Dan Deacon", "Tocotronic", "Wir Sind Helden") :
    nodeMusic.addNode(node)
print("DONE!")
print()

print("### ADDING SECOND LEVEL CHILDREN...", end = "")
nodeMusic[0].addNode("dummy")

for node in ("Die Reklamation", "Von Hier An Blind", "Soundso", "Bring Mich Nach Hause") :
    nodeMusic[-1].addNode(node)
print("DONE!")
print()



print("### MANUAL OUTPUT ROOT LEVEL")
for sub in tree.subtrees :
    print(sub)
print()

print("### MANUAL OUTPUT FIRST LEVEL UNDER NODE DOCUMENTS")
for sub in nodeDocs.subtrees :
    print(sub)
print()

print("### MANUAL OUTPUT FIRST LEVEL UNDER NODE EBOOKS")
for sub in tree[1].subtrees :
    print(sub)
print()

print("### MANUAL OUTPUT FIRST LEVEL UNDER NODE MUSIC")
for sub in tree[-2].subtrees :
    print(sub)
print()



print("### REMOVING A SUBTREE...", end = "")
nodeMusic.removeNode(0)
print("DONE!")
print()


print("### ITERATOR OUTPUT OF FULL TREE")
for indent, item in tree :
    print("  " * indent, item, sep="")