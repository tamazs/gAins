// Validates that the response has all required top-level fields with correct types
module.exports = (output) => {
  try {
    const clean = output.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
      .replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
    const d = JSON.parse(clean);
    return (
      typeof d.overall_summary === 'string' && d.overall_summary.length > 10 &&
      Array.isArray(d.exercise_advice) && d.exercise_advice.length >= 1 &&
      d.exercise_advice.every(ea =>
        typeof ea.exercise_name === 'string' &&
        typeof ea.recommendation === 'string' &&
        typeof ea.reasoning === 'string'
      ) &&
      typeof d.recovery_flag === 'boolean' &&
      Array.isArray(d.sources_used)
    );
  } catch (e) {
    return false;
  }
};
