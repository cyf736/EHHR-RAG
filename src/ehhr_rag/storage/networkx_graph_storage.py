import html
from pathlib import Path
from typing import Any, Union, cast

import networkx as nx

from ehhr_rag.config import dataset_db_dir
from ehhr_rag.io_utils import create_file_if_not_exists
from ehhr_rag.logging_utils import logger


class NetworkXStorage:
    @staticmethod
    def load_nx_graph(file_name: str | Path) -> nx.Graph | None:
        file_name = Path(file_name)
        if file_name.exists():
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name: str | Path) -> None:
        logger.info("Writing graph with %s nodes, %s edges", graph.number_of_nodes(), graph.number_of_edges())
        create_file_if_not_exists(file_name)
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
        sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[0])
        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))
        if not graph.is_directed():
            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    source, target = target, source
                return source, target, edge_data
            edges = [_sort_source_target(edge) for edge in edges]
        edges = sorted(edges, key=lambda x: f"{x[0]} -> {x[1]}")
        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __init__(self, namespace: str, base_dir: str | Path | None = None):
        self.namespace = namespace
        resolved_base_dir = Path(base_dir) if base_dir is not None else dataset_db_dir()
        self._graphml_xml_file = resolved_base_dir / f"graph_{self.namespace}.graphml"
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                "Loaded graph from %s with %s nodes, %s edges",
                self._graphml_xml_file,
                preloaded_graph.number_of_nodes(),
                preloaded_graph.number_of_edges(),
            )
        self._graph = preloaded_graph or nx.Graph()

    def return_self_graph(self):
        return self._graph

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree[node_id]

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_edges(self):
        return self._graph.edges()

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str):
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info("Node %s deleted from the graph.", node_id)
        else:
            logger.warning("Node %s not found in the graph for deletion.", node_id)

    async def get_subgraph(self, nodes: list) -> nx.Graph:
        return self._graph.subgraph(nodes).copy()

    async def get_neighbors(self, _id: str):
        try:
            return list(self._graph.neighbors(_id))
        except Exception:
            return []

    async def find_nodes_by_prefix(self, prefix):
        return [node for node in self._graph.nodes if str(node).startswith(prefix)]
