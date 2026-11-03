import ts from "typescript";

// Parsing TypeScript keeps quotes inside template expressions from truncating a route.
export function clientCalls(source) {
  const tree = ts.createSourceFile("client.ts", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const calls = [];
  function visit(node) {
    if (ts.isCallExpression(node) && node.expression.getText(tree) === "api") {
      const [target, options] = node.arguments;
      let path;
      if (target && ts.isStringLiteralLike(target)) path = target.text;
      else if (target && ts.isTemplateExpression(target)) {
        path = target.head.text;
        for (const span of target.templateSpans) {
          if (path.includes("?")) break;
          const expression = span.expression;
          if (ts.isCallExpression(expression) && expression.expression.getText(tree) === "queryString") break;
          if (ts.isConditionalExpression(expression)) {
            const branches = [expression.whenTrue, expression.whenFalse];
            if (branches.every((item) => ts.isStringLiteralLike(item) && (!item.text || item.text.startsWith("?")))) break;
          }
          path += "{}" + span.literal.text;
        }
      }
      if (path !== undefined) {
        const isJsonBody = (value) => ts.isCallExpression(value) && value.expression.getText(tree) === "jsonBody";
        let method = options && isJsonBody(options) ? "post" : "get";
        if (options && ts.isObjectLiteralExpression(options)) {
          for (const property of options.properties) {
            if (ts.isSpreadAssignment(property) && isJsonBody(property.expression)) method = "post";
            if (property.name?.getText(tree) === "jsonBody") method = "post";
            if (ts.isPropertyAssignment(property) && property.name.getText(tree) === "method" && ts.isStringLiteralLike(property.initializer)) {
              method = property.initializer.text.toLowerCase();
            }
          }
        }
        calls.push({ path: path.replace(/\?.*$/, ""), method, rawPath: target.getText(tree) });
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(tree);
  return calls;
}

export function contractErrors(calls, spec) {
  const shapes = new Map(Object.entries(spec.paths).map(([path, operations]) => [
    path.replace(/\{[^}]*\}/g, "{}"), new Set(Object.keys(operations)),
  ]));
  return calls.flatMap(({ path, method, rawPath }) => {
    const methods = shapes.get(path);
    if (!methods) return [`MISSING ${method.toUpperCase()} ${rawPath}`];
    if (!methods.has(method)) return [`METHOD ${method.toUpperCase()} ${rawPath} (backend: ${[...methods].join("/")})`];
    return [];
  });
}
