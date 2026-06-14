// A small mutable RGB color (components in 0..1), with hex helpers. Games
// commonly set colors as 0xRRGGBB and mutate them in place
// (material.color.setHex(...)), so this mirrors that ergonomic surface.

export class Color {
  constructor(hexOrR = 0xffffff, g, b) {
    if (g === undefined) this.setHex(hexOrR);
    else { this.r = hexOrR; this.g = g; this.b = b; }
  }

  setHex(hex) {
    this.r = ((hex >> 16) & 0xff) / 255;
    this.g = ((hex >> 8) & 0xff) / 255;
    this.b = (hex & 0xff) / 255;
    return this;
  }

  set(r, g, b) { this.r = r; this.g = g; this.b = b; return this; }
  setRGB(r, g, b) { return this.set(r, g, b); }
  copy(c) { this.r = c.r; this.g = c.g; this.b = c.b; return this; }
  clone() { return new Color(this.r, this.g, this.b); }
  toArray() { return [this.r, this.g, this.b]; }
}
