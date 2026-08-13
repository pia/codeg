"use client"

import { useCallback, useEffect, useState } from "react"
import { Download, Languages, Trash2 } from "lucide-react"
import { useTranslations } from "next-intl"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { toErrorMessage } from "@/lib/app-error"
import {
  deleteReasoningTranslationModel,
  downloadReasoningTranslationModel,
  getReasoningTranslationModelStatus,
  getReasoningTranslationSettings,
  updateReasoningTranslationSettings,
} from "@/lib/api"
import { invalidateReasoningTranslationSettingsCache } from "@/lib/reasoning-translation"
import type {
  ReasoningTranslationSettings,
  TranslationModelStatus,
} from "@/lib/types"

const STATUS_POLL_MS = 1000

function formatMegabytes(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1)
}

export function ReasoningTranslationSettingsSection() {
  const t = useTranslations("GeneralSettings.reasoningTranslation")
  const [settings, setSettings] = useState<ReasoningTranslationSettings | null>(
    null
  )
  const [status, setStatus] = useState<TranslationModelStatus | null>(null)
  const [saving, setSaving] = useState(false)
  const [acting, setActing] = useState(false)
  const [downloadStarted, setDownloadStarted] = useState(false)

  const load = useCallback(async () => {
    try {
      const [nextSettings, nextStatus] = await Promise.all([
        getReasoningTranslationSettings(),
        getReasoningTranslationModelStatus(),
      ])
      setSettings(nextSettings)
      setStatus(nextStatus)
    } catch (err) {
      toast.error(t("loadFailed", { message: toErrorMessage(err) }))
    }
  }, [t])

  useEffect(() => {
    void load()
  }, [load])

  // Poll after a download has been kicked off until it settles. The kickoff
  // command returns the state at the moment the task was spawned, which is
  // usually still `not_downloaded`; waiting only for `downloading` would
  // therefore miss both fast completions and the transition itself.
  useEffect(() => {
    if (!downloadStarted) return
    const state = status?.state
    if (state !== "not_downloaded" && state !== "downloading") {
      setDownloadStarted(false)
      return
    }
    const timer = window.setInterval(() => {
      getReasoningTranslationModelStatus()
        .then((next) => {
          setStatus(next)
          if (next.state !== "not_downloaded" && next.state !== "downloading") {
            setDownloadStarted(false)
          }
        })
        .catch(() => {
          // Polling is best-effort; the next tick retries.
        })
    }, STATUS_POLL_MS)
    return () => window.clearInterval(timer)
  }, [downloadStarted, status?.state])

  const onEnabledChange = useCallback(
    async (enabled: boolean) => {
      if (!settings) return
      setSaving(true)
      try {
        const next = await updateReasoningTranslationSettings({
          ...settings,
          enabled,
        })
        setSettings(next)
        invalidateReasoningTranslationSettingsCache()
      } catch (err) {
        toast.error(t("saveFailed", { message: toErrorMessage(err) }))
      } finally {
        setSaving(false)
      }
    },
    [settings, t]
  )

  const onDownload = useCallback(async () => {
    setActing(true)
    try {
      setStatus(await downloadReasoningTranslationModel())
      setDownloadStarted(true)
    } catch (err) {
      toast.error(t("downloadFailed", { message: toErrorMessage(err) }))
    } finally {
      setActing(false)
    }
  }, [t])

  const onDelete = useCallback(async () => {
    if (!window.confirm(t("deleteModelConfirm"))) return
    setActing(true)
    try {
      setStatus(await deleteReasoningTranslationModel())
    } catch (err) {
      toast.error(t("deleteFailed", { message: toErrorMessage(err) }))
    } finally {
      setActing(false)
    }
  }, [t])

  const statusLabel = (() => {
    switch (status?.state) {
      case "downloading":
        return status.total_bytes !== null
          ? t("modelDownloading", {
              downloaded: formatMegabytes(status.downloaded_bytes),
              total: formatMegabytes(status.total_bytes),
            })
          : t("modelDownloadingUnknownTotal", {
              downloaded: formatMegabytes(status.downloaded_bytes),
            })
      case "ready":
        return t("modelReady")
      case "failed":
        return t("modelFailed", { message: status.message })
      default:
        return t("modelNotDownloaded")
    }
  })()

  const downloading = status?.state === "downloading"

  return (
    <section className="rounded-xl border bg-card p-4 space-y-4">
      <div className="flex items-center gap-2">
        <Languages className="h-4 w-4 text-muted-foreground" />
        <h2 className="text-sm font-semibold">{t("sectionTitle")}</h2>
      </div>

      <p className="text-xs text-muted-foreground leading-5">
        {t("sectionDescription")}
      </p>

      <label className="flex items-center gap-2">
        <Switch
          checked={settings?.enabled ?? false}
          onCheckedChange={(checked) => void onEnabledChange(checked)}
          disabled={saving || !settings}
        />
        <span className="text-xs text-muted-foreground">{t("enabled")}</span>
      </label>

      <div className="space-y-2">
        <p className="text-[11px] text-muted-foreground">{statusLabel}</p>
        <div className="flex items-center gap-2">
          {status?.state !== "ready" && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => void onDownload()}
              disabled={acting || downloading}
            >
              <Download className="h-3.5 w-3.5" />
              {t("downloadModel")}
            </Button>
          )}
          {status?.state === "ready" && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => void onDelete()}
              disabled={acting}
            >
              <Trash2 className="h-3.5 w-3.5" />
              {t("deleteModel")}
            </Button>
          )}
        </div>
      </div>

      <p className="text-[11px] text-muted-foreground/80 leading-5">
        {t("modelInfo")}
      </p>
    </section>
  )
}
