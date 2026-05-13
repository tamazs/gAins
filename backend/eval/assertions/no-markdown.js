// Checks that output does not start with markdown code fences
// (the app parses the response directly as JSON)
module.exports = (output) => {
  const stripped = output.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  return !stripped.startsWith('```');
};
