import { paretoFront } from "./pareto";
import assert from "node:assert";

const P = (id: string, latency: number | null, map: number | null, valid = true) => ({ id, latency, map, valid });
assert.deepStrictEqual(paretoFront([P("a", 10, 0.3), P("b", 20, 0.35), P("c", 30, 0.3)]).sort(), ["a", "b"]);
assert.deepStrictEqual(paretoFront([P("a", 10, 0.3), P("t", 10, 0.3)]).sort(), ["a", "t"], "ties kept");
assert.deepStrictEqual(paretoFront([P("x", 1, 0.99, false), P("n", 1, null), P("ok", 5, 0.2)]), ["ok"]);
assert.deepStrictEqual(paretoFront([]), []);
console.log("pareto.ts tests passed");
