// Stay still inside the safe area; move only near an edge. Distant jumps cut.
export function cameraTarget(center, head, width, height, teleport = false) {
  const outside = Math.abs(head[0] - center[0]) > (width - 1) / 2 ||
    Math.abs(head[1] - center[1]) > (height - 1) / 2;
  if (teleport && outside) return {
    center: [...head],
    cut: true
  };
  const next = center.map((value, axis) => {
    const size = axis ? height : width;
    const half = (size - 1) / 2;
    const margin = Math.min(2, Math.floor(half));
    const distance = head[axis] - value;
    return Math.abs(distance) > half - margin ?
      head[axis] - Math.sign(distance) * Math.floor(half / 2) : value;
  });
  return {
    center: next,
    cut: outside && teleport
  };
}

export function validDimension(value) {
  const number = Number(value);
  return Number.isInteger(number) && number >= 3 && number <= 101;
}
