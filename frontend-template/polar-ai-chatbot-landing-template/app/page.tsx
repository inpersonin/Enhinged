"use client"

import { useState } from "react"
import { BrandBadge } from "@/components/polar/brand-badge"
import { ChatCard } from "@/components/polar/chat-card"
import { FooterLinks } from "@/components/polar/footer-links"

const DEFAULT_BACKGROUND = "/images/background.png"

export default function PolarLandingPage() {
  const [backgroundImage, setBackgroundImage] = useState(DEFAULT_BACKGROUND)

  const handleResetBackground = () => {
    setBackgroundImage(DEFAULT_BACKGROUND)
  }

  return (
    <main
      className="relative flex min-h-screen flex-col items-center justify-between bg-cover bg-center bg-no-repeat px-4 py-8"
      style={{
        backgroundImage: `url('${backgroundImage}')`,
      }}
    >
      {/* Gradient overlay for better contrast */}
      <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent" />

      {/* Content */}
      <div className="relative z-10 flex w-full flex-col items-center pt-8">
        <BrandBadge />
      </div>

      <div className="relative z-10 flex w-full flex-col items-center">
        <ChatCard userName="Juan" onBackgroundChange={setBackgroundImage} onResetBackground={handleResetBackground} />
      </div>

      <div className="relative z-10 pb-4">
        <FooterLinks />
      </div>
    </main>
  )
}
