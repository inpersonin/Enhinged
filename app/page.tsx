import { Navbar } from "@/components/navbar"
import { ChatHero } from "@/components/chat-hero"
import { CustomCursor } from "@/components/custom-cursor"
import { SmoothScroll } from "@/components/smooth-scroll"

export default function Home() {
  return (
    <SmoothScroll>
      <CustomCursor />
      <Navbar />
      <main>
        <ChatHero />
      </main>
    </SmoothScroll>
  )
}