import { SnowflakeIcon } from "@/components/icons/snowflake-icon"

export function BrandBadge() {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/5 px-4 py-2 backdrop-blur-md">
      <SnowflakeIcon className="h-5 w-5 text-sky-400" />
      <span className="text-sm font-medium text-white">Polar.ai</span>
    </div>
  )
}
