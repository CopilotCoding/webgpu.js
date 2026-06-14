// A GeometryArena packs many distinct meshes (which all share one vertex
// layout) into a few large GPU buffers so they can be drawn together with
// indexed indirect draws — the prerequisite for GPU-driven rendering of
// HETEROGENEOUS geometry. Each mesh is sub-allocated a slice of the shared
// vertex and index buffers; a draw references it by { baseVertex, firstIndex,
// indexCount }.
//
// All meshes in one arena use the same interleaved vertex format. The default
// is the engine's standard layout — position(float32x3), normal(float32x3),
// uv(float32x2) = 32 bytes/vertex — but any stride works as long as every
// mesh matches it.
//
// Allocation is bump + free-list. free() returns a slice to the list; a later
// allocate() reuses the smallest fitting free slice. The buffers grow by
// allocating a larger GPU buffer and copying the old contents on the GPU
// (copyBufferToBuffer) — no CPU readback.

const DEFAULT_VERTEX_STRIDE = 32; // pos(12) + normal(12) + uv(8)

export class GeometryArena {
  /**
   * @param {object} device webgpu.js Device
   * @param {object} [opts]
   * @param {number} [opts.vertexStride=32] bytes per vertex
   * @param {GPUVertexBufferLayout[]} [opts.vertexBufferLayouts] layout for pipelines drawing from this arena
   * @param {number} [opts.initialVertices=65536]
   * @param {number} [opts.initialIndices=131072]
   * @param {'uint32'} [opts.indexFormat='uint32']
   */
  constructor(device, opts = {}) {
    this.device = device;
    this.vertexStride = opts.vertexStride ?? DEFAULT_VERTEX_STRIDE;
    this.indexFormat = opts.indexFormat ?? 'uint32';
    this.indexSize = this.indexFormat === 'uint16' ? 2 : 4;

    // Default interleaved layout: position/normal/uv.
    this.vertexBufferLayouts = opts.vertexBufferLayouts ?? [{
      arrayStride: this.vertexStride,
      stepMode: 'vertex',
      attributes: [
        { format: 'float32x3', offset: 0, shaderLocation: 0 },
        { format: 'float32x3', offset: 12, shaderLocation: 1 },
        { format: 'float32x2', offset: 24, shaderLocation: 2 },
      ],
    }];

    this.vertexCapacity = opts.initialVertices ?? 65536; // in vertices
    this.indexCapacity = opts.initialIndices ?? 131072;  // in indices
    this.vertexHead = 0; // next free vertex (bump)
    this.indexHead = 0;  // next free index (bump)

    this.vertexBuffer = this._makeVertexBuffer(this.vertexCapacity);
    this.indexBuffer = this._makeIndexBuffer(this.indexCapacity);

    // Free lists: arrays of { offset, count } (in vertices / indices).
    this._freeVertices = [];
    this._freeIndices = [];
  }

  _makeVertexBuffer(capacityVertices) {
    return this.device.resources.createBuffer({
      size: capacityVertices * this.vertexStride,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  _makeIndexBuffer(capacityIndices) {
    return this.device.resources.createBuffer({
      size: capacityIndices * this.indexSize,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  _allocFrom(freeList, head, count, capacity, grow) {
    // Try the smallest fitting free slice first.
    let bestIdx = -1;
    for (let i = 0; i < freeList.length; i++) {
      if (freeList[i].count >= count && (bestIdx === -1 || freeList[i].count < freeList[bestIdx].count)) {
        bestIdx = i;
      }
    }
    if (bestIdx !== -1) {
      const slice = freeList[bestIdx];
      const offset = slice.offset;
      if (slice.count === count) freeList.splice(bestIdx, 1);
      else { slice.offset += count; slice.count -= count; }
      return { offset, head, capacity };
    }
    // Bump-allocate, growing the buffer if needed.
    let newHead = head + count;
    let newCapacity = capacity;
    while (newHead > newCapacity) newCapacity *= 2;
    if (newCapacity !== capacity) grow(newCapacity);
    return { offset: head, head: newHead, capacity: newCapacity };
  }

  _growVertices(newCapacity) {
    const next = this._makeVertexBuffer(newCapacity);
    const encoder = this.device.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.vertexBuffer.gpuBuffer, 0, next.gpuBuffer, 0, this.vertexHead * this.vertexStride);
    this.device.queue.submit([encoder.finish()]);
    this.vertexBuffer.destroy();
    this.vertexBuffer = next;
    this.vertexCapacity = newCapacity;
  }

  _growIndices(newCapacity) {
    const next = this._makeIndexBuffer(newCapacity);
    const encoder = this.device.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.indexBuffer.gpuBuffer, 0, next.gpuBuffer, 0, this.indexHead * this.indexSize);
    this.device.queue.submit([encoder.finish()]);
    this.indexBuffer.destroy();
    this.indexBuffer = next;
    this.indexCapacity = newCapacity;
  }

  /**
   * Uploads one mesh into the arena.
   * @param {Float32Array} vertexData interleaved, matching vertexStride
   * @param {Uint32Array|Uint16Array} indexData
   * @returns {{ baseVertex:number, firstIndex:number, indexCount:number, vertexCount:number, handle:object }}
   */
  allocate(vertexData, indexData) {
    const vertexCount = (vertexData.byteLength) / this.vertexStride;
    const indexCount = indexData.length;

    const v = this._allocFrom(this._freeVertices, this.vertexHead, vertexCount, this.vertexCapacity, (c) => this._growVertices(c));
    if (v.head !== undefined) this.vertexHead = v.head;
    const baseVertex = v.offset;

    const idx = this._allocFrom(this._freeIndices, this.indexHead, indexCount, this.indexCapacity, (c) => this._growIndices(c));
    if (idx.head !== undefined) this.indexHead = idx.head;
    const firstIndex = idx.offset;

    this.device.queue.writeBuffer(this.vertexBuffer.gpuBuffer, baseVertex * this.vertexStride, vertexData);
    const indices = this.indexFormat === 'uint16' ? Uint16Array.from(indexData) : Uint32Array.from(indexData);
    this.device.queue.writeBuffer(this.indexBuffer.gpuBuffer, firstIndex * this.indexSize, indices);

    const handle = { baseVertex, firstIndex, indexCount, vertexCount };
    return { baseVertex, firstIndex, indexCount, vertexCount, handle };
  }

  /** Returns a mesh's slices to the free lists for reuse. */
  free(handle) {
    this._freeVertices.push({ offset: handle.baseVertex, count: handle.vertexCount });
    this._freeIndices.push({ offset: handle.firstIndex, count: handle.indexCount });
  }

  /** Binds the arena's vertex + index buffers into an active render pass. */
  bind(renderPass) {
    renderPass.setVertexBuffer(0, this.vertexBuffer.gpuBuffer);
    renderPass.setIndexBuffer(this.indexBuffer.gpuBuffer, this.indexFormat);
  }

  destroy() {
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
  }
}

/**
 * Packs separate position/normal/uv arrays (the shape `primitives.js`
 * generators return) into one interleaved Float32Array matching the arena's
 * default 32-byte layout, and a trivial 0..N index buffer (the generators are
 * non-indexed triangle soup).
 */
export function interleaveStandard(data) {
  const vertexCount = data.positions.length / 3;
  const out = new Float32Array(vertexCount * 8);
  const hasUV = !!data.uvs;
  for (let i = 0; i < vertexCount; i++) {
    out[i * 8 + 0] = data.positions[i * 3 + 0];
    out[i * 8 + 1] = data.positions[i * 3 + 1];
    out[i * 8 + 2] = data.positions[i * 3 + 2];
    out[i * 8 + 3] = data.normals[i * 3 + 0];
    out[i * 8 + 4] = data.normals[i * 3 + 1];
    out[i * 8 + 5] = data.normals[i * 3 + 2];
    out[i * 8 + 6] = hasUV ? data.uvs[i * 2 + 0] : 0;
    out[i * 8 + 7] = hasUV ? data.uvs[i * 2 + 1] : 0;
  }
  const indices = new Uint32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) indices[i] = i;
  return { vertexData: out, indexData: indices };
}
