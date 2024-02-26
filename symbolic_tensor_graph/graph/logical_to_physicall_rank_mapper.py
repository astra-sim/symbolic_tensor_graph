from functools import lru_cache
import copy
import math

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
    def factored_generate_mappings(cls, factored_logical, physical):
        mappings = list()
        for mapping in cls._factored_mappings_generator(factored_logical, physical):
            phy_to_logical = dict()
            for (logical_idx, logical_elm, logical_factor_idx, logical_factor_elm), physical_idx in mapping.items():
                if not physical_idx in phy_to_logical:
                    phy_to_logical[physical_idx] = dict()
                if not logical_idx in phy_to_logical[physical_idx]:
                    phy_to_logical[physical_idx][logical_idx] = logical_factor_elm
                else:
                    phy_to_logical[physical_idx][logical_idx] *= logical_factor_elm
            if not phy_to_logical in mappings:
                mappings.append(phy_to_logical)
        return mappings

    @classmethod
    def generate_mappings(cls, logical, physical):
        assert math.prod(logical) == math.prod(physical)    
        group_factors_logical = cls.get_group_factors(logical)
        mappings = cls.factored_generate_mappings(group_factors_logical, physical)
        return mappings


if __name__ == '__main__':
    logical = [32, 8, 4, 4]
    physical = [64, 8, 8]
    
    mappings = LogicalToPhysicalRankMapper.generate_mappings(logical, physical)
    print(mappings)
    print(len(mappings))
    