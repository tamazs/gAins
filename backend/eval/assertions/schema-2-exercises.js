// Schema compliance requiring at least 2 exercise_advice entries
module.exports = (output) => {
  try {
    const clean = output.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
      .replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
    const d = JSON.parse(clean);
    return (
      typeof d.overall_summary === 'string' && d.overall_summary.length > 10 &&
      Array.isArray(d.exercise_advice) && d.exercise_advice.length >= 2 &&
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
