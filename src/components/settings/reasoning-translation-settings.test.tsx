import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { NextIntlClientProvider } from "next-intl"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ReasoningTranslationSettingsSection } from "./reasoning-translation-settings"
import * as api from "@/lib/api"

vi.mock("@/lib/api", () => ({
  getReasoningTranslationSettings: vi.fn(),
  updateReasoningTranslationSettings: vi.fn(),
  getReasoningTranslationModelStatus: vi.fn(),
  downloadReasoningTranslationModel: vi.fn(),
  deleteReasoningTranslationModel: vi.fn(),
}))

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), message: vi.fn() },
}))

function renderSection() {
  return render(
    <NextIntlClientProvider
      locale="zh-CN"
      messages={{
        GeneralSettings: {
          reasoningTranslation: {
            sectionTitle: "思维链翻译",
            sectionDescription: "描述",
            enabled: "启用",
            modelNotDownloaded: "未下载",
            modelDownloading: "下载中 {downloaded} MB / {total} MB",
            modelDownloadingUnknownTotal: "下载中 {downloaded} MB",
            modelReady: "已就绪",
            modelFailed: "失败：{message}",
            downloadModel: "下载模型",
            deleteModel: "删除模型",
            deleteModelConfirm: "确认删除？",
            modelInfo: "模型信息",
            saveFailed: "保存失败：{message}",
            loadFailed: "加载失败：{message}",
            downloadFailed: "下载失败：{message}",
            deleteFailed: "删除失败：{message}",
          },
        },
      }}
    >
      <ReasoningTranslationSettingsSection />
    </NextIntlClientProvider>
  )
}

beforeEach(() => {
  vi.mocked(api.getReasoningTranslationSettings).mockResolvedValue({
    enabled: false,
    target_language: "zh-CN",
  })
  vi.mocked(api.getReasoningTranslationModelStatus).mockResolvedValue({
    state: "not_downloaded",
  })
})

describe("ReasoningTranslationSettingsSection", () => {
  it("shows disabled default and persists the toggle", async () => {
    vi.mocked(api.updateReasoningTranslationSettings).mockResolvedValue({
      enabled: true,
      target_language: "zh-CN",
    })
    renderSection()

    expect(await screen.findByText("思维链翻译")).toBeTruthy()
    const toggle = screen.getByRole("switch")
    expect(toggle).toHaveAttribute("aria-checked", "false")

    fireEvent.click(toggle)
    await waitFor(() =>
      expect(api.updateReasoningTranslationSettings).toHaveBeenCalledWith({
        enabled: true,
        target_language: "zh-CN",
      })
    )
  })

  it("shows delete action for a ready model and confirms before deleting", async () => {
    vi.mocked(api.getReasoningTranslationModelStatus).mockResolvedValue({
      state: "ready",
      revision: "rev",
    })
    vi.mocked(api.deleteReasoningTranslationModel).mockResolvedValue({
      state: "not_downloaded",
    })
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true)
    renderSection()

    fireEvent.click(await screen.findByText("删除模型"))
    await waitFor(() =>
      expect(api.deleteReasoningTranslationModel).toHaveBeenCalled()
    )
    expect(confirm).toHaveBeenCalled()
  })
})
