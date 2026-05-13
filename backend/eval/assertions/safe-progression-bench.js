// Verifies suggested bench press weight does not exceed 10% above current 100kg baseline
module.exports = (output) => {
  try {
    const clean = output.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
      .replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
    const d = JSON.parse(clean);
    const bench = d.exercise_advice.find(ea =>
      ea.exercise_name.toLowerCase().includes('bench')
    );
    if (!bench || bench.suggested_weight_kg == null) return true;
    return bench.suggested_weight_kg <= 110;
  } catch (e) {
    return false;
  }
};
