"use client"

import { useEffect, useRef, useState } from "react"

import {
  getReasoningTranslationSettings,
  translateReasoningSegments,
} from "@/lib/api"
import type { ReasoningTranslationSettings } from "@/lib/types"

export interface ReasoningSegment {
  kind: "prose" | "code" | "inline-code" | "url"
  text: string
}

const TERMINATORS = new Set([".", "!", "?", "。", "！", "？"])
const URL_RE = /https?:\/\/[^\s`]+/g
const WINDOWS_PATH_RE = /[A-Za-z]:\\[^\s`]+/g
const UNIX_PATH_RE = /\/[\w.+-]+(?:\/[\w.+-]+)+/g

/** Small LRU cache for translated prose segments (memory-only). */
export class ReasoningTranslationCache {
  private readonly map = new Map<string, string>()

  constructor(private readonly capacity = 500) {}

  get(key: string): string | undefined {
    const value = this.map.get(key)
    if (value !== undefined) {
      this.map.delete(key)
      this.map.set(key, value)
    }
    return value
  }

  set(key: string, value: string): void {
    if (this.map.has(key)) this.map.delete(key)
    this.map.set(key, value)
    if (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value
      if (oldest !== undefined) this.map.delete(oldest)
    }
  }
}

const proseCache = new ReasoningTranslationCache()

function pushProse(segments: ReasoningSegment[], text: string): void {
  if (!text) return
  let pos = 0
  for (const match of text.matchAll(URL_RE)) {
    const start = match.index
    if (start > pos) pushPaths(segments, text.slice(pos, start))
    segments.push({ kind: "url", text: match[0] })
    pos = start + match[0].length
  }
  if (pos < text.length) pushPaths(segments, text.slice(pos))
}

function pushPaths(segments: ReasoningSegment[], text: string): void {
  if (!text) return
  let pos = 0
  const combined = new RegExp(
    `(${WINDOWS_PATH_RE.source})|(${UNIX_PATH_RE.source})`,
    "g"
  )
  for (const match of text.matchAll(combined)) {
    const start = match.index
    if (start > pos)
      segments.push({ kind: "prose", text: text.slice(pos, start) })
    segments.push({ kind: "url", text: match[0] })
    pos = start + match[0].length
  }
  if (pos < text.length) segments.push({ kind: "prose", text: text.slice(pos) })
}

function pushInline(segments: ReasoningSegment[], text: string): void {
  if (!text) return
  let pos = 0
  while (pos < text.length) {
    const tick = text.indexOf("`", pos)
    if (tick === -1) {
      pushProse(segments, text.slice(pos))
      break
    }
    pushProse(segments, text.slice(pos, tick))
    const end = text.indexOf("`", tick + 1)
    if (end === -1) {
      pushProse(segments, text.slice(tick))
      break
    }
    segments.push({ kind: "inline-code", text: text.slice(tick, end + 1) })
    pos = end + 1
  }
}

/**
 * Split markdown into prose/code/inline-code/url segments. Joining all
 * segment texts reproduces the input exactly.
 */
export function splitMarkdownPreservingCode(
  markdown: string
): ReasoningSegment[] {
  const segments: ReasoningSegment[] = []
  let pos = 0
  while (pos < markdown.length) {
    const fenceStart = markdown.indexOf("```", pos)
    if (fenceStart === -1) {
      pushInline(segments, markdown.slice(pos))
      break
    }
    pushInline(segments, markdown.slice(pos, fenceStart))
    const fenceEnd = markdown.indexOf("```", fenceStart + 3)
    if (fenceEnd === -1) {
      segments.push({ kind: "code", text: markdown.slice(fenceStart) })
      break
    }
    segments.push({
      kind: "code",
      text: markdown.slice(fenceStart, fenceEnd + 3),
    })
    pos = fenceEnd + 3
  }
  return segments
}

/**
 * Split text at sentence terminators. The final element may be an incomplete
 * trailing sentence (no terminator), which callers should keep untranslated
 * while streaming.
 */
export function splitIntoSentences(text: string): string[] {
  const sentences: string[] = []
  let current = ""
  for (const ch of text) {
    current += ch
    if (TERMINATORS.has(ch)) {
      sentences.push(current)
      current = ""
    }
  }
  if (current) sentences.push(current)
  return sentences.length > 0 ? sentences : [text]
}

export function isCompleteSentence(sentence: string): boolean {
  return sentence.length > 0 && TERMINATORS.has(sentence[sentence.length - 1])
}

let settingsPromise: Promise<ReasoningTranslationSettings> | null = null

function getSettingsCached(): Promise<ReasoningTranslationSettings> {
  settingsPromise ??= getReasoningTranslationSettings().catch((err) => {
    settingsPromise = null
    throw err
  })
  return settingsPromise
}

/** Settings page calls this after saving so blocks pick up the change. */
export function invalidateReasoningTranslationSettingsCache(): void {
  settingsPromise = null
}

interface TranslationResult {
  displayText: string
  view: "translation" | "original"
  isPending: boolean
  setView: (view: "translation" | "original") => void
  translationEnabled: boolean
}

/**
 * Display-layer translation for one reasoning block:
 * - complete block: all prose segments are translated together;
 * - streaming block: only complete sentences are translated, the trailing
 *   incomplete sentence stays original until it completes;
 * - results are cached per prose segment in memory.
 */
export function useReasoningTranslation(
  content: string,
  isStreaming: boolean
): TranslationResult {
  const [view, setView] = useState<"translation" | "original">("translation")
  const [translated, setTranslated] = useState<string | null>(null)
  const [isPending, setIsPending] = useState(false)
  const [translationEnabled, setTranslationEnabled] = useState(false)
  const generationRef = useRef(0)

  useEffect(() => {
    let alive = true
    getSettingsCached()
      .then((settings) => {
        if (alive) setTranslationEnabled(settings.enabled)
      })
      .catch(() => {
        // Settings unavailable: fall back to original text.
      })
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    if (!translationEnabled) {
      const timer = window.setTimeout(() => {
        setTranslated(null)
        setIsPending(false)
      }, 0)
      return () => window.clearTimeout(timer)
    }

    const generation = ++generationRef.current
    const segments = splitMarkdownPreservingCode(content)
    const proseSegments = segments.filter((s) => s.kind === "prose")
    const pendingTexts: string[] = []
    const seen = new Set<string>()

    for (const segment of proseSegments) {
      const sentences = splitIntoSentences(segment.text)
      const toTranslate = isStreaming
        ? sentences.filter(isCompleteSentence)
        : sentences
      for (const sentence of toTranslate) {
        if (!seen.has(sentence) && proseCache.get(sentence) === undefined) {
          seen.add(sentence)
          pendingTexts.push(sentence)
        }
      }
    }

    const rebuild = () => {
      if (generation !== generationRef.current) return
      const next = segments
        .map((segment) => {
          if (segment.kind !== "prose") return segment.text
          const cached = proseCache.get(segment.text)
          if (cached !== undefined) return cached
          return splitIntoSentences(segment.text)
            .map((sentence) => proseCache.get(sentence) ?? sentence)
            .join("")
        })
        .join("")
      setTranslated(next)
    }

    const timer = window.setTimeout(
      () => {
        if (pendingTexts.length === 0) {
          setIsPending(false)
          rebuild()
          return
        }
        setIsPending(true)
        translateReasoningSegments(pendingTexts)
          .then((translations) => {
            translations.forEach((translation, index) => {
              proseCache.set(pendingTexts[index], translation)
            })
            if (generation === generationRef.current) {
              setIsPending(false)
              rebuild()
            }
          })
          .catch(() => {
            if (generation === generationRef.current) {
              setIsPending(false)
              rebuild()
            }
          })
      },
      isStreaming ? 250 : 0
    )

    return () => window.clearTimeout(timer)
  }, [content, isStreaming, translationEnabled])

  const displayText =
    view === "translation" && translated !== null ? translated : content

  return {
    displayText,
    view,
    isPending,
    setView,
    translationEnabled,
  }
}
