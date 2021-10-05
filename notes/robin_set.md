# Robin Set

Robin set is a high-performing set that uses the robin hood hashing method defined in tsl.h

The core concept behind this data structure is to take from "rich" values and give back to "poor" values - very much like the tale and hence, the name.

This is achieved by replacing the values with low **psl** (probe sequence length) with values having higher psl.

This is realised by modifying the insertion order. 

> **Insertion invariant:** all values that hash to bucket $i$ precede all values that hash to bucket $(i+1)$ for any $i$.

```cpp
void insert(v) {
    p = hash(v) % length;
    while(arr[p] != null) {
        if(vpsl > b[p].psl) {
            swap(v, b[p].v);
            swap(vpsl, b[p].psl);
        }

        p = p+1 % length;
        vpsl++;
    }
    b[p].v = v;
    b[p].psl = vpsl;
}
```

**Searching:** required to search until one of the three conditions:
1. b[p] = null
2. b[p].v = v
3. vpsl > b[p].psl

Source: [Cornell - Robin Hashing](https://www.cs.cornell.edu/courses/JavaAndDS/files/hashing_RobinHood.pdf)