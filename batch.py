import torch
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_sparse import SparseTensor, cat


class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ["edge_index", "masked_atom_indices"]:
                    item = item + cumsum_node
                elif key == "connected_edge_indices":
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].cat_dim(key, batch[key][0])
            )
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in [
            "edge_index",
            "face",
            "masked_atom_indices",
            "connected_edge_indices",
        ]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchAE(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchAE, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ["edge_index", "negative_edge_index"]:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0


class BatchSubstructContext(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        # keys = [set(data.keys) for data in data_list]
        # keys = list(set.union(*keys))
        # assert 'batch' not in keys

        batch = BatchSubstructContext()
        keys = [
            "center_substruct_idx",
            "edge_attr_substruct",
            "edge_index_substruct",
            "x_substruct",
            "overlap_context_substruct_idx",
            "edge_attr_context",
            "edge_index_context",
            "x_context",
        ]

        for key in keys:
            # print(key)
            batch[key] = []

        # batch.batch = []
        # used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0

        for data in data_list:
            # If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                # batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(
                    torch.full(
                        (len(data.overlap_context_substruct_idx),), i, dtype=torch.long
                    )
                )
                batch.overlapped_context_size.append(
                    len(data.overlap_context_substruct_idx)
                )

                # batching for the main graph
                # for key in data.keys:
                #    if not "context" in key and not "substruct" in key:
                #        item = data[key]
                #        item = item + cumsum_main if batch.cumsum(key, item) else item
                #        batch[key].append(item)

                # batching for the substructure graph
                for key in [
                    "center_substruct_idx",
                    "edge_attr_substruct",
                    "edge_index_substruct",
                    "x_substruct",
                ]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)

                # batching for the context graph
                for key in [
                    "overlap_context_substruct_idx",
                    "edge_attr_context",
                    "edge_index_context",
                    "x_context",
                ]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.cat_dim(key))
        # batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(
            batch.batch_overlapped_context, dim=-1
        )
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        return batch.contiguous()

    def cat_dim(self, key):
        return (
            -1
            if key in ["edge_index", "edge_index_substruct", "edge_index_context"]
            else 0
        )

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in [
            "edge_index",
            "edge_index_substruct",
            "edge_index_context",
            "overlap_context_substruct_idx",
            "center_substruct_idx",
        ]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchPubchemContext(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        # keys = [set(data.keys) for data in data_list]
        # keys = list(set.union(*keys))
        # assert 'batch' not in keys

        batch = BatchSubstructContext()
        keys = [
            "center_substruct_idx",
            "edge_attr_substruct",
            "edge_index_substruct",
            "x_substruct",
            "overlap_context_substruct_idx",
            "edge_attr_context",
            "edge_index_context",
            "x_context",
        ]

        for key in keys:
            # print(key)
            batch[key] = []

        # batch.batch = []
        # used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []
        # used for pooling the substruct
        batch.batch_center_substruct = []
        batch.center_substruct_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0

        for data in data_list:
            # If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                # batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(
                    torch.full(
                        (len(data.overlap_context_substruct_idx),), i, dtype=torch.long
                    )
                )
                batch.overlapped_context_size.append(
                    len(data.overlap_context_substruct_idx)
                )
                batch.batch_center_substruct.append(
                    torch.full((len(data.center_substruct_idx),), i, dtype=torch.long)
                )
                batch.center_substruct_size.append(len(data.center_substruct_idx))

                # batching for the main graph
                # for key in data.keys:
                #    if not "context" in key and not "substruct" in key:
                #        item = data[key]
                #        item = item + cumsum_main if batch.cumsum(key, item) else item
                #        batch[key].append(item)

                # batching for the substructure graph
                for key in [
                    "center_substruct_idx",
                    "edge_attr_substruct",
                    "edge_index_substruct",
                    "x_substruct",
                ]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)

                # batching for the context graph
                for key in [
                    "overlap_context_substruct_idx",
                    "edge_attr_context",
                    "edge_index_context",
                    "x_context",
                ]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.cat_dim(key))
        # batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(
            batch.batch_overlapped_context, dim=-1
        )
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)
        batch.batch_center_substruct = torch.cat(batch.batch_center_substruct, dim=-1)
        batch.center_substruct_size = torch.LongTensor(batch.center_substruct_size)

        return batch.contiguous()

    def cat_dim(self, key):
        return (
            -1
            if key in ["edge_index", "edge_index_substruct", "edge_index_context"]
            else 0
        )

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in [
            "edge_index",
            "edge_index_substruct",
            "edge_index_context",
            "overlap_context_substruct_idx",
            "center_substruct_idx",
        ]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchPoolingIndicator(Batch):
    """ overwrite the original from_data_list method in pyg.data.Batch to handle pooling
    indicators.
    """

    @classmethod
    def from_data_list(cls, data_list, follow_batch=[], exclude_keys=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""

        keys = list(set(data_list[0].keys) - set(exclude_keys))
        assert "batch" not in keys and "ptr" not in keys

        batch = cls()
        for key in data_list[0].__dict__.keys():
            if key[:2] != "__" and key[-2:] != "__":
                batch[key] = None

        batch.__num_graphs__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ["batch"]:
            batch[key] = []
        batch["ptr"] = [0]

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        indicator_cum = 0
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if key == "pooling_indicator":
                    if i == 0:
                        batch["emb_indicator"] = torch.zeros(
                            (data["pooling_indicator"][-1].item() + 1, 1),
                            dtype=torch.long,
                        )
                    else:
                        batch["emb_indicator"] = torch.cat(
                            [
                                batch["emb_indicator"],
                                torch.ones(
                                    (data["pooling_indicator"][-1].item() + 1, 1),
                                    dtype=torch.long,
                                )
                                * i,
                            ],
                            dim=0,
                        )
                    temp = data["pooling_indicator"][-1].item() + 1
                    item += indicator_cum
                    indicator_cum += temp
                elif isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value + cum
                        item = item.set_value(value, layout="coo")
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                # 0-dimensional tensors have no dimension along which to
                # concatenate, so we set `cat_dim` to `None`.
                if isinstance(item, Tensor) and item.dim() == 0:
                    cat_dim = None
                cat_dims[key] = cat_dim

                # Add a batch dimension to items whose `cat_dim` is `None`:
                if isinstance(item, Tensor) and cat_dim is None:
                    cat_dim = 0  # Concatenate along this new batch dimension.
                    item = item.unsqueeze(0)
                    device = item.device
                elif isinstance(item, Tensor):
                    size = item.size(cat_dim)
                    device = item.device
                elif isinstance(item, SparseTensor):
                    size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                    device = item.device()

                batch[key].append(item)  # Append item to the attribute list.

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f"{key}_{j}_batch"
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size,), i, dtype=torch.long, device=device)
                            )
                    else:
                        tmp = f"{key}_batch"
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size,), i, dtype=torch.long, device=device)
                        )

            if hasattr(data, "__num_nodes__"):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes,), i, dtype=torch.long, device=device)
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_nodes)

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        for key in batch.keys:
            if key == "emb_indicator":
                continue
            items = batch[key]
            item = items[0]
            cat_dim = ref_data.__cat_dim__(key, item)
            cat_dim = 0 if cat_dim is None else cat_dim
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, cat_dim)
            elif isinstance(item, SparseTensor):
                batch[key] = cat(items, cat_dim)
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()
