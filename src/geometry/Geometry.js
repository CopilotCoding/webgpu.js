import { VERTEX_FORMAT_SIZES, INDEX_FORMAT_SIZES } from './vertexFormats.js';

/**
 * Immutable geometry: one GPU buffer per named vertex attribute, plus an
 * optional index buffer. Each attribute's data is uploaded once at creation
 * and never modified — dynamic geometry is a separate, explicit path.
 *
 * descriptor shape:
 * {
 *   attributes: {
 *     position: { format: 'float32x3', data: Float32Array, stepMode?: 'vertex' | 'instance' },
 *     normal:   { format: 'float32x3', data: Float32Array },
 *     ...
 *   },
 *   indices: { format: 'uint16' | 'uint32', data: Uint16Array | Uint32Array },
 * }
 */
export class Geometry {
  constructor(device, descriptor) {
    this.attributes = {};
    this.vertexBufferLayouts = [];
    this.index = null;

    let shaderLocation = 0;
    for (const [name, attribute] of Object.entries(descriptor.attributes ?? {})) {
      const formatSize = VERTEX_FORMAT_SIZES[attribute.format];
      if (formatSize === undefined) {
        throw new Error(`Geometry: unknown GPUVertexFormat "${attribute.format}" for attribute "${name}"`);
      }

      const buffer = device.resources.createBuffer({
        size: attribute.data.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer.gpuBuffer, 0, attribute.data);

      const location = shaderLocation++;

      this.attributes[name] = { buffer, format: attribute.format, shaderLocation: location, vertexCount: attribute.data.length / (formatSize / 4), data: attribute.data };

      this.vertexBufferLayouts.push({
        arrayStride: formatSize,
        stepMode: attribute.stepMode ?? 'vertex',
        attributes: [{ format: attribute.format, offset: 0, shaderLocation: location }],
      });
    }

    if (descriptor.indices) {
      const indexFormatSize = INDEX_FORMAT_SIZES[descriptor.indices.format];
      if (indexFormatSize === undefined) {
        throw new Error(`Geometry: unknown GPUIndexFormat "${descriptor.indices.format}"`);
      }

      const buffer = device.resources.createBuffer({
        size: descriptor.indices.data.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer.gpuBuffer, 0, descriptor.indices.data);

      this.index = { buffer, format: descriptor.indices.format, count: descriptor.indices.data.length };
    }
  }

  get vertexCount() {
    const first = Object.values(this.attributes)[0];
    return first ? first.vertexCount : 0;
  }

  /** Computes a local-space AABB from the "position" attribute's data. */
  computeBounds() {
    const position = this.attributes.position;
    if (!position) {
      throw new Error('Geometry.computeBounds: geometry has no "position" attribute');
    }

    const data = position.data;
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];

    for (let i = 0; i < data.length; i += 3) {
      for (let axis = 0; axis < 3; axis++) {
        const value = data[i + axis];
        if (value < min[axis]) min[axis] = value;
        if (value > max[axis]) max[axis] = value;
      }
    }

    return { min, max };
  }

  destroy() {
    for (const attribute of Object.values(this.attributes)) {
      attribute.buffer.destroy();
    }
    if (this.index) this.index.buffer.destroy();
  }
}
