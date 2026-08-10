import { describe, expect, it } from "vitest"

import {
  ReasoningTranslationCache,
  isCompleteSentence,
  splitIntoSentences,
  splitMarkdownPreservingCode,
} from "./reasoning-translation"

describe("splitMarkdownPreservingCode", () => {
  it("keeps fenced code and inline code verbatim", () => {
    const markdown =
      "Let me check.\n```ts\nconst x = 1;\n```\nThen `x` is used."
    const parts = splitMarkdownPreservingCode(markdown)
    expect(parts.filter((p) => p.kind === "code")).toHaveLength(1)
    expect(parts.find((p) => p.kind === "code")?.text).toContain("const x = 1;")
    expect(parts.filter((p) => p.kind === "inline-code")).toHaveLength(1)
    expect(parts.map((p) => p.text).join("")).toBe(markdown)
  })

  it("keeps urls and windows paths out of prose", () => {
    const markdown = "See https://example.com/x and C:\\tmp\\a.txt now."
    const parts = splitMarkdownPreservingCode(markdown)
    expect(parts.filter((p) => p.kind === "url").map((p) => p.text)).toEqual([
      "https://example.com/x",
      "C:\\tmp\\a.txt",
    ])
    expect(parts.map((p) => p.text).join("")).toBe(markdown)
  })
})

describe("splitIntoSentences", () => {
  it("splits at terminators and marks the tail incomplete", () => {
    const sentences = splitIntoSentences("One. Two! Three? Still writing")
    // Leading whitespace stays attached to the following sentence so that
    // joining all sentences reconstructs the original text exactly.
    expect(sentences).toEqual(["One.", " Two!", " Three?", " Still writing"])
    expect(sentences.slice(0, 3).every(isCompleteSentence)).toBe(true)
    expect(isCompleteSentence(sentences[3])).toBe(false)
  })
})

describe("ReasoningTranslationCache", () => {
  it("evicts oldest entries beyond capacity", () => {
    const cache = new ReasoningTranslationCache(2)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")
    expect(cache.get("a")).toBeUndefined()
    expect(cache.get("b")).toBe("2")
    expect(cache.get("c")).toBe("3")
  })
})
