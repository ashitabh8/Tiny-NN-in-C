"""
Parallel region detection for diamond-shaped subgraphs.

A parallel region is a fan-out/fan-in diamond created by
QuantizationTransform._insert_parallel_branch() (e.g. LQER error correction):

    source --+--> main[0] --> ... --> main[-1] --+
             |                                   v
             +--> branch[0] --> ... --> branch[-1] --> join

The two chains are topologically independent, so backends may execute them
concurrently (the C printer emits OpenMP sections) and the memory planner
must treat their buffers as simultaneously live.

Detection is metadata-based so this module only depends on the IR:
- join node:    metadata['parallel_join'] = True, inputs = [main_out, branch_out]
- branch nodes: metadata['parallel_branch'] = True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..ir.node import IRNode


@dataclass
class ParallelRegion:
    """One fan-out/fan-in diamond. Chains are in execution order."""

    join: IRNode
    source: IRNode
    main_chain: List[IRNode] = field(default_factory=list)
    branch_chain: List[IRNode] = field(default_factory=list)

    @property
    def node_names(self) -> List[str]:
        """Names of all nodes inside the region (excluding source and join)."""
        return [n.name for n in self.main_chain + self.branch_chain]


def find_parallel_regions(order: List[IRNode]) -> List[ParallelRegion]:
    """
    Find all parallel regions in an execution order.

    Walks back from each join node: the branch chain is the run of
    'parallel_branch' nodes feeding the join's second input; the node feeding
    the branch head is the shared source; the main chain is the linear chain
    between the source and the join's first input.
    """
    regions: List[ParallelRegion] = []
    in_order = {n.name for n in order}

    for node in order:
        if not (node.metadata and node.metadata.get('parallel_join')):
            continue
        if len(node.inputs) != 2:
            continue

        main_out, branch_out = node.inputs[0], node.inputs[1]

        # Branch chain: walk back through tagged branch nodes.
        branch_chain: List[IRNode] = []
        cur = branch_out
        while (cur is not None
               and cur.metadata
               and cur.metadata.get('parallel_branch')):
            branch_chain.append(cur)
            cur = cur.inputs[0] if cur.inputs else None
        if not branch_chain or cur is None:
            continue  # branch was removed/rewired by a later pass; no region
        source = cur
        branch_chain.reverse()

        # Main chain: linear walk from the join's first input back to source.
        main_chain: List[IRNode] = []
        node_walk = main_out
        ok = True
        for _ in range(len(order)):
            if node_walk is source:
                break
            main_chain.append(node_walk)
            if not node_walk.inputs:
                ok = False
                break
            node_walk = node_walk.inputs[0]
        else:
            ok = False
        if not ok:
            continue
        main_chain.reverse()

        # All region nodes must be present in the given order.
        if any(n.name not in in_order for n in main_chain + branch_chain):
            continue

        regions.append(ParallelRegion(
            join=node,
            source=source,
            main_chain=main_chain,
            branch_chain=branch_chain,
        ))

    return regions


def region_index_spans(
    order: List[IRNode],
    regions: List[ParallelRegion],
) -> Dict[str, int]:
    """
    Map every node name inside a region to the order-index of its join.

    Used by the memory planner: any buffer defined or last-used inside a
    parallel region must stay live until the join executes, because the two
    chains run concurrently and have no defined relative order.
    """
    order_index = {n.name: i for i, n in enumerate(order)}
    span: Dict[str, int] = {}
    for region in regions:
        join_idx = order_index.get(region.join.name)
        if join_idx is None:
            continue
        for name in region.node_names:
            span[name] = join_idx
    return span
