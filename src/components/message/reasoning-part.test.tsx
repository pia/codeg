import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { NextIntlClientProvider } from "next-intl"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useReasoningTranslation } from "@/lib/reasoning-translation"

vi.mock("@/lib/api", () => ({
  getReasoningTranslationSettings: vi.fn(),
  translateReasoningSegments: vi.fn(),
}))

import * as api from "@/lib/api"

function Harness({
  content,
  isStreaming,
}: {
  content: string
  isStreaming: boolean
}) {
  const { displayText, view, isPending, setView, translationEnabled } =
    useReasoningTranslation(content, isStreaming)
  return (
    <div>
      <span data-testid="text">{displayText}</span>
      <span data-testid="view">{view}</span>
      <span data-testid="enabled">{String(translationEnabled)}</span>
      <span data-testid="pending">{String(isPending)}</span>
      <button
        type="button"
        onClick={() =>
          setView(view === "translation" ? "original" : "translation")
        }
      >
        toggle
      </button>
    </div>
  )
}

beforeEach(() => {
  vi.mocked(api.getReasoningTranslationSettings).mockResolvedValue({
    enabled: true,
    target_language: "zh-CN",
  })
  vi.mocked(api.translateReasoningSegments).mockImplementation(
    async (segments: string[]) => segments.map((s) => `译:${s}`)
  )
})

describe("useReasoningTranslation", () => {
  it("translates a complete block and can toggle back to original", async () => {
    render(
      <NextIntlClientProvider locale="zh-CN" messages={{}}>
        <Harness content="Hello world." isStreaming={false} />
      </NextIntlClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId("enabled").textContent).toBe("true")
    })
    await waitFor(() => {
      expect(screen.getByTestId("text").textContent).toBe("译:Hello world.")
    })

    fireEvent.click(screen.getByText("toggle"))
    expect(screen.getByTestId("view").textContent).toBe("original")
    expect(screen.getByTestId("text").textContent).toBe("Hello world.")
  })

  it("keeps the incomplete trailing sentence in english while streaming", async () => {
    render(
      <NextIntlClientProvider locale="zh-CN" messages={{}}>
        <Harness
          content="First complete sentence. Second incomplete"
          isStreaming={true}
        />
      </NextIntlClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId("text").textContent).toContain(
        "译:First complete sentence."
      )
    })
    expect(screen.getByTestId("text").textContent).toContain(
      "Second incomplete"
    )
  })
})
