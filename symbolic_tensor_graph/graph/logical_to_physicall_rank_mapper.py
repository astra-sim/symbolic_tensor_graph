from functools import lru_cache
import copy
import math
from collections import OrderedDict

class LogicalToPhysicalRankMapper:
    _prime_numbers = list()
    _prime_number_limit = 0
    
    @classmethod
    def get_prime_numbers(cls, number_limit=2048):
        assert number_limit < 1e6, "avoid taking too much memory"
        if number_limit <= cls._prime_number_limit:
            return cls._prime_numbers
        primes = [True] * (number_limit+1)
        primes[0] = False
        primes[1] = False
        p = 2
        while p*p <= number_limit:
            if primes[p]:
                for i in range(p*p, number_limit+1, p):
                    primes[i] = False
            p += 1
        primes = [i for i in range(number_limit+1) if primes[i]]
        cls._prime_numbers = primes
        cls._prime_number_limit = number_limit
        return cls._prime_numbers
        
    @classmethod
    @lru_cache
    def get_prime_factors(cls, number):
        number = int(number)
        prime_numbers = cls.get_prime_numbers(number)
        prime_factors = list()
        for current_try in prime_numbers:
            while number % current_try == 0:
                prime_factors.append(current_try)
                number = number // current_try
            if number == 1:
                break
        assert number == 1
        return prime_factors
    
    @classmethod
    def get_group_factors(cls, group):
        # tuple: (group_idx, group_element, factor_idx, factor_element)
        group_factors = list()
        for group_idx, group_element in enumerate(group):
            factors = cls.get_prime_factors(group_element)
            for factor_idx, factor_element in enumerate(factors):
                group_factors.append((group_idx, group_element, factor_idx, factor_element))
        return group_factors
    
    @classmethod
    def _factored_mappings_generator(cls, factored_logical, physical, mapping=None, idx=0):
        if idx == len(factored_logical):
            # print("yield!")
            yield mapping
            return
        physical = tuple(physical)
        if mapping is None:
            mapping = dict()
        factor = factored_logical[idx]
        for i, physical_dim in enumerate(physical):
            if not physical_dim % factor[3] == 0:
                continue
            # print(f"alloc {factor} = {i} with mapping={mapping}, factored_logical={factored_logical[idx:]}")
            mapping[factor] = i
            next_physical = physical[:i]+(physical[i]//factor[3],)+physical[i+1:]
            yield from cls._factored_mappings_generator(factored_logical, next_physical, mapping, idx+1)
            del mapping[factor]
        return
    
    @classmethod
    def factored_generate_logical_to_phy_mappings(cls, factored_logical, physical):
        logical_to_phy_mappings = list()
        for mapping in cls._factored_mappings_generator(factored_logical, physical):
            logical_to_phy = dict()
            for (logical_idx, logical_elm, logical_factor_idx, logical_factor_elm), physical_idx in mapping.items():
                if not logical_idx in logical_to_phy:
                    logical_to_phy[logical_idx] = dict()
                if not physical_idx in logical_to_phy[logical_idx]:
                    logical_to_phy[logical_idx][physical_idx] = logical_factor_elm
                else:
                    logical_to_phy[logical_idx][physical_idx] *= logical_factor_elm
            if not logical_to_phy in logical_to_phy_mappings:
                logical_to_phy_mappings.append(logical_to_phy)
        return logical_to_phy_mappings

    @classmethod
    def generate_logical_to_phy_mappings(cls, logical, physical):
        assert math.prod(logical) == math.prod(physical)    
        group_factors_logical = cls.get_group_factors(logical)
        logical_to_phy_mappings = cls.factored_generate_logical_to_phy_mappings(group_factors_logical, physical)
        return logical_to_phy_mappings
    
    @classmethod
    def create_addr_map_rank(cls, space_shape):
        physical_addr_map_rank = dict()
        for rank in range(math.prod(space_shape)):
            addr = list(space_shape)
            for i, _ in enumerate(addr):
                addr[i] = -1
            leftover = rank
            denominator = math.prod(space_shape)
            for i, dim in enumerate(reversed(space_shape)):
                denominator = denominator//dim
                quotient = leftover // denominator
                leftover %= denominator
                addr[i] = quotient
            addr = tuple(reversed(addr))
            physical_addr_map_rank[addr] = rank
        return physical_addr_map_rank
    
    @classmethod
    def logical_to_phy_mapping_to_readable_rank_map_number_rank(cls, mapping, logical, physical):
        def _complete_logical_to_phy_mapping(mapping_, logical_, physical_):
            completed = dict()
            for i in range(len(logical_)):
                if not i in mapping_.keys():
                    completed[i] = dict()
                else:
                    completed[i] = mapping_[i]
                for j in range(len(physical_)):
                    if not j in completed[i]:
                        completed[i][j] = 1
            return completed
        mapping = _complete_logical_to_phy_mapping(mapping, logical, physical)
        def _create_accululated_base(sliced_mapping_):
            accumulated_base = dict()
            for logical_dim in range(len(sliced_mapping_)):
                assert logical_dim in sliced_mapping_
                accumulated_base[logical_dim] = dict()
                for physical_dim in sliced_mapping_[logical_dim]:
                    if logical_dim == 0:
                        accumulated_base[logical_dim][physical_dim] = 1
                    else:
                        accumulated_base[logical_dim][physical_dim] = \
                            accumulated_base[logical_dim-1][physical_dim] * sliced_mapping_[logical_dim-1][physical_dim]
            return accumulated_base
        accumulated_base = _create_accululated_base(mapping)
        def _expand_logical_to_phy_mapping(sliced_mapping_, accumulcated_base, logical_addr_):
            physical_addr = list()
            for _ in range(len(sliced_mapping_[0].keys())):
                physical_addr.append(0)
            for logical_dim in range(len(sliced_mapping_)):
                physical_sub_addr = list()
                remain = logical_addr_[logical_dim]
                for physical_dim in range(len(sliced_mapping_[logical_dim])):
                    addr_this_dim = remain % sliced_mapping_[logical_dim][physical_dim]
                    remain = remain // sliced_mapping_[logical_dim][physical_dim]
                    physical_sub_addr.append(addr_this_dim)
                    physical_addr[physical_dim] += addr_this_dim * accumulcated_base[logical_dim][physical_dim]
            return tuple(physical_addr)
        logical_addrs_map_rank = cls.create_addr_map_rank(logical)
        logical_addrs_map_physical_addrs = dict()
        logical_addrs_map_physical_rank = dict()
        physical_addr_map_physical_rank = cls.create_addr_map_rank(physical)
        for logical_addr in logical_addrs_map_rank.keys():
            physical_addr = _expand_logical_to_phy_mapping(mapping, accumulated_base, logical_addr)
            logical_addrs_map_physical_addrs[logical_addr] = tuple(physical_addr)
            logical_addrs_map_physical_rank[logical_addr] = physical_addr_map_physical_rank[physical_addr]
        return logical_addrs_map_physical_rank
    
    @classmethod
    def generate_all_readable_mappings(cls, readable_ranks, physical):
        logical_dimensions = OrderedDict()
        for dimension, dimension_rank in readable_ranks[0]:
            logical_dimensions[dimension] = 0
        for readable_rank in readable_ranks:
            for dimension, dimension_rank in readable_rank:
                logical_dimensions[dimension] = max(logical_dimensions[dimension], dimension_rank)
        for dimension in logical_dimensions.keys():
            logical_dimensions[dimension] += 1
        
        def _logical_add_dimension(logical_values_, logical_dimension_):
            readable = list()
            for value, dim in zip(logical_values_, logical_dimension_.keys()):
                readable.append((dim, value))
            return tuple(readable)
        
        logical = list()
        for _, size in logical_dimensions.items():
            logical.append(size)
        mappings = cls.generate_logical_to_phy_mappings(logical, physical)
        expanded_mappings = list()
        for mapping in mappings:
            expanded_mapping = cls.logical_to_phy_mapping_to_readable_rank_map_number_rank(mapping, logical, physical)
            readable_mapping = dict()
            for key, value in expanded_mapping.items():
                readable_key = _logical_add_dimension(key, logical_dimensions)
                readable_mapping[readable_key] = value
            expanded_mappings.append(readable_mapping)
        return expanded_mappings, mappings


if __name__ == '__main__':
    # logical = [1, 32, 32]
    physical = [128, 2, 4]
    import pickle
    f = open("readable_rank_8448.pkl", "rb")
    readable_ranks = pickle.load(f)
    f.close()
    expanded_mappings, mappings = LogicalToPhysicalRankMapper.generate_all_readable_mappings(readable_ranks, physical)
    hook = 0
    # mappings = LogicalToPhysicalRankMapper.generate_logical_to_phy_mappings(logical, physical)
    # print(mappings)
    # print(len(mappings))
    # LogicalToPhysicalRankMapper.logical_to_phy_mapping_to_readable_rank_map_number_rank(mappings[1], logical, physical)
    