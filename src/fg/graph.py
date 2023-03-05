from typing import List, Dict


class Node:
    def __init__(self, name: str, dims: list) -> None:
        self._name = name
        self._dims = dims
        self._graph: Graph = None

    def is_type(self, _class) -> bool:
        return type(self) == _class

    def __str__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        return self._name

    @property
    def dims(self) -> list:
        return self._dims.copy()

    @property
    def neighbors(self) -> List['Node']:
        if self._graph is None:
            return []
        return self._graph.neighbors(self)

    @property
    def edges(self) -> List['Edge']:
        if self._graph is None:
            return []
        return self._graph._node_edges[self].copy()


class Edge:
    def __init__(self, node0: Node, node1: Node) -> None:
        self._node0: Node = node0
        self._node1: Node = node1
        self._messages = {}

    def disconnect(self):
        self._node0

    def get_other(self, node: Node) -> Node:
        if self._node0 is node:
            return self._node1
        elif self._node1 is node:
            return self._node0
        raise 'Node not connected with this edge'

    def set_message_from(self, node: Node, value):
        self._set_message(node, value)

    def get_message_to(self, node: Node):
        return self._get_message(self.get_other(node))

    def _set_message(self, key, value):
        self._messages[key] = value

    def _get_message(self, key):
        return self._messages.get(key, None)

class Graph:
    def __init__(self) -> None:
        self._nodes: List[Node] = []
        self._edges: List[Edge] = []
        self._node_edges: Dict[List[Edge]] = {}

    def add_node(self, node: Node):
        if node in self._nodes:
            return
        node._graph = self
        self._nodes.append(node)
        self._node_edges[node] = []
    def remove_node(self, node: Node):
        if node not in self._nodes:
            return
        self._nodes.remove(node)
        edges = self._node_edges.pop(node)
        for e in edges:
            self._edges.remove(e)
            o: Node = e.get_other(node)
            if o in self._node_edges:
                self._node_edges[o].remove(e)

    def connect(self, node0: Node, node1: Node) -> Edge:
        if node0 not in self._nodes:
            self.add_node(node0)
        if node1 not in self._nodes:
            self.add_node(node1)
        for e in self._node_edges[node0]:
            if e.get_other(node0) is node1:
                return
        for e in self._node_edges[node1]:
            if e.get_other(node1) is node0:
                return
        e = Edge(node0, node1)
        self._edges.append(e)
        self._node_edges[node0].append(e)
        self._node_edges[node1].append(e)

    def connect_seq(self, *seq: List[Node]):
        for i in range(len(seq)-1):
            self.connect(seq[i], seq[i+1])

    def disconnect(self, node0: Node, node1: Node) -> Edge:
        if node0 not in self._nodes or node1 not in self._nodes:
            return None

        edge = None
        for e in self._node_edges[node0]:
            if e.get_other(node0) is node1:
                edge = e
                break
        if edge is None:
            return None
        self._edges.remove(edge)
        self._node_edges[node0].remove(edge)
        self._node_edges[node1].remove(edge)

    def remove_edge(self, edge: Edge):
        if edge not in self._edges:
            return
        self._edges.remove(edge)
        if edge._node0 in self._node_edges:
            self._node_edges[edge._node0].remove(edge)
        if edge._node1 in self._node_edges:
            self._node_edges[edge._node1].remove(edge)

    def neighbors(self, node: Node) -> List[Node]:
        if node not in self._nodes:
            return []
        return [e.get_other(node) for e in self._node_edges[node]]


if __name__ == '__main__':
    node0 = Node('n0', [1, 2])
    node1 = Node('n1', [1, 2])
    g = Graph()
    g.connect(node0, node1)
    g.disconnect(node0, node1)
    print(g._edges)

