# Reproduce, capture and replay

A strict JSON request contains only case, resolution_scale and seed:

```json
{"case":"multilayer-coating","resolution_scale":1,"seed":42}
```

Resolution is an integer from 1 to 3. Higher values preserve physical dimensions
while increasing spatial/time samples; sparse 2D cost grows quickly.
Seeds are integers in [0,2^32). Unknown fields, boolean integers and duplicate
batch case names are rejected.

```bash
emlab run --config examples/coating.json --output outputs/coating
emlab batch --data-only --output outputs/catalog
emlab render outputs/catalog
emlab verify outputs/catalog
```

Batch progress is written after each completed study. An interrupted partial catalog
contains its completed studies; it is not automatically advertised as a full gallery.

## Archive contents

fields.npz contains finite numeric arrays and a Unicode JSON metadata record.
Loading disables pickle. Complex fields remain complex arrays; metadata uses finite
JSON values. Explicit boolean masks identify singular, absorbing or unilluminated
regions rather than replacing them with misleading plotted values.

Metadata records the request, actual physical parameters, assumptions, tables,
plot specification and package/environment versions. Git revision and dirty status
are captured before the physical calculation. Configuration and physical parameters
have separate SHA-256 digests.

Each array is hashed over dtype, shape and C-order bytes. The metadata has its own
digest. This seals correspondence among fields, setup and reported summaries.

## Replay and verification

```bash
emlab replay outputs/coating/fields.npz --output outputs/replayed
emlab verify outputs/replayed
```

Replay uses the stored arrays and plot specification. It preserves the original
solver source metadata, including dirty status; it never relabels an old capture
as a fresh solve. Rendering software can produce different image or NPZ container
bytes across versions while preserving sealed array values and source metadata.

A flat manifest hashes every report file. Verification checks safe filenames,
file hashes, archive seals, and agreement between exported JSON and archive metadata.
Manifests are unsigned integrity records and can be rewritten by an author.
They establish neither authenticity nor physical correctness.

Generated text uses LF line endings so checked-in manifests survive cross-platform
Git checkout. The published gallery requires all 24 default requests from one clean
recorded source revision. Its later documentation commit is expected to differ from
the revision that performed the calculations.
