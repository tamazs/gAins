// Checks that recovery_flag is true (overtraining signals present)
module.exports = (output) => {
  try {
    const clean = output.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
      .replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
    return JSON.parse(clean).recovery_flag === true;
  } catch (e) {
    return false;
  }
};
