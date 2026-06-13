// Procedural texture layers drawn with the 2D canvas API, returned as an
// array of canvases the Engine uploads into its albedo texture array:
// 0 brick, 1 lit-window facade, 2 concrete panels, 3 pavement.
export function makeTextureLayers(size) {
  return [0, 1, 2, 3].map((layer) => {
    const c = document.createElement('canvas');
    c.width = size;
    c.height = size;
    draw(c.getContext('2d'), layer, size);
    return c;
  });
}

function draw(ctx, layer, size) {
  if (layer === 0) {
    ctx.fillStyle = '#9a4a3a';
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#c0c0b8';
    const rows = 8, brickH = size / rows;
    for (let r = 0; r < rows; r++) {
      ctx.fillRect(0, r * brickH - 2, size, 4);
      const offset = (r % 2) * (size / 8);
      for (let c = 0; c < 4; c++) ctx.fillRect(((c * size) / 4 + offset) % size - 2, r * brickH, 4, brickH);
    }
  } else if (layer === 1) {
    ctx.fillStyle = '#23262e';
    ctx.fillRect(0, 0, size, size);
    const cells = 8, cell = size / cells;
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        ctx.fillStyle = (x * 31 + y * 17) % 7 < 3 ? '#ffd98a' : '#11131a';
        ctx.fillRect(x * cell + cell * 0.2, y * cell + cell * 0.2, cell * 0.6, cell * 0.6);
      }
    }
  } else if (layer === 2) {
    ctx.fillStyle = '#8d8d92';
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = '#6b6b70';
    ctx.lineWidth = 3;
    const panels = 4, panel = size / panels;
    for (let i = 0; i <= panels; i++) {
      ctx.beginPath(); ctx.moveTo(i * panel, 0); ctx.lineTo(i * panel, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * panel); ctx.lineTo(size, i * panel); ctx.stroke();
    }
    ctx.fillStyle = 'rgba(0,0,0,0.12)';
    for (let i = 0; i < 60; i++) ctx.fillRect((i * 37) % size, (i * 91) % size, 6, 6);
  } else {
    const cells = 8, cell = size / cells;
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        ctx.fillStyle = (x + y) % 2 === 0 ? '#7c7f86' : '#5e6168';
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }
    }
    ctx.strokeStyle = '#46484e';
    ctx.lineWidth = 2;
    for (let i = 0; i <= cells; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(size, i * cell); ctx.stroke();
    }
  }
}
