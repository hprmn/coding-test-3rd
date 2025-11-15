'use client'

import { useQuery } from '@tanstack/react-query'
import { useParams } from 'next/navigation'
import { fundApi } from '@/lib/api'
import { formatCurrency, formatPercentage, formatDate } from '@/lib/utils'
import { Loader2, TrendingUp, DollarSign, Calendar, Download } from 'lucide-react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function FundDetailPage() {
  const params = useParams()
  const fundId = parseInt(params.id as string)

  const { data: fund, isLoading: fundLoading } = useQuery({
    queryKey: ['fund', fundId],
    queryFn: () => fundApi.get(fundId)
  })

  const { data: capitalCalls } = useQuery({
    queryKey: ['transactions', fundId, 'capital_calls'],
    queryFn: () => fundApi.getTransactions(fundId, 'capital_calls', 1, 10)
  })

  const { data: distributions } = useQuery({
    queryKey: ['transactions', fundId, 'distributions'],
    queryFn: () => fundApi.getTransactions(fundId, 'distributions', 1, 10)
  })

  const { data: allCapitalCalls } = useQuery({
    queryKey: ['transactions', fundId, 'capital_calls', 'all'],
    queryFn: () => fundApi.getTransactions(fundId, 'capital_calls', 1, 100)
  })

  const { data: allDistributions } = useQuery({
    queryKey: ['transactions', fundId, 'distributions', 'all'],
    queryFn: () => fundApi.getTransactions(fundId, 'distributions', 1, 100)
  })

  if (fundLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    )
  }

  if (!fund) {
    return <div>Fund not found</div>
  }

  const metrics = fund.metrics || {}

  // Prepare chart data
  const cashFlowData = prepareCashFlowData(allCapitalCalls?.items || [], allDistributions?.items || [])
  const cumulativeData = prepareCumulativeData(allCapitalCalls?.items || [], allDistributions?.items || [])

  const handleExportExcel = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/funds/${fundId}/export`
      )

      if (!response.ok) throw new Error('Export failed')

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${fund.name}_Export.xlsx`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Export failed:', error)
      alert('Failed to export data')
    }
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8 flex items-start justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2">{fund.name}</h1>
          <div className="flex items-center space-x-4 text-gray-600">
            {fund.gp_name && <span>GP: {fund.gp_name}</span>}
            {fund.vintage_year && <span>Vintage: {fund.vintage_year}</span>}
            {fund.fund_type && <span>Type: {fund.fund_type}</span>}
          </div>
        </div>
        <button
          onClick={handleExportExcel}
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition flex items-center space-x-2"
        >
          <Download className="w-5 h-5" />
          <span>Export to Excel</span>
        </button>
      </div>

      {/* Metrics Cards */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="DPI"
          value={metrics.dpi?.toFixed(2) + 'x' || 'N/A'}
          description="Distribution to Paid-In"
          icon={<TrendingUp className="w-6 h-6" />}
          color="blue"
        />
        <MetricCard
          title="IRR"
          value={metrics.irr ? formatPercentage(metrics.irr) : 'N/A'}
          description="Internal Rate of Return"
          icon={<TrendingUp className="w-6 h-6" />}
          color="green"
        />
        <MetricCard
          title="Paid-In Capital"
          value={metrics.pic ? formatCurrency(metrics.pic) : 'N/A'}
          description="Total capital called"
          icon={<DollarSign className="w-6 h-6" />}
          color="purple"
        />
        <MetricCard
          title="Distributions"
          value={metrics.total_distributions ? formatCurrency(metrics.total_distributions) : 'N/A'}
          description="Total distributions"
          icon={<DollarSign className="w-6 h-6" />}
          color="orange"
        />
      </div>

      {/* Charts Section */}
      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        {/* Cash Flow Chart */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Cash Flow Timeline</h2>
          {cashFlowData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={cashFlowData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip
                  formatter={(value: number) => formatCurrency(Math.abs(value))}
                  labelFormatter={(label) => `Date: ${label}`}
                />
                <Legend />
                <Bar dataKey="capitalCalls" fill="#ef4444" name="Capital Calls" />
                <Bar dataKey="distributions" fill="#10b981" name="Distributions" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-gray-500 text-sm text-center py-20">No cash flow data available</p>
          )}
        </div>

        {/* Cumulative Performance Chart */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Cumulative Performance</h2>
          {cumulativeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={cumulativeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip
                  formatter={(value: number) => formatCurrency(value)}
                  labelFormatter={(label) => `Date: ${label}`}
                />
                <Legend />
                <Line type="monotone" dataKey="cumulativeCalls" stroke="#ef4444" strokeWidth={2} name="Cumulative Calls" />
                <Line type="monotone" dataKey="cumulativeDistributions" stroke="#10b981" strokeWidth={2} name="Cumulative Distributions" />
                <Line type="monotone" dataKey="netCashFlow" stroke="#3b82f6" strokeWidth={2} name="Net Cash Flow" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-gray-500 text-sm text-center py-20">No performance data available</p>
          )}
        </div>
      </div>

      {/* Transactions Tables */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Capital Calls */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Capital Calls</h2>
          {capitalCalls && capitalCalls.items.length > 0 ? (
            <div className="space-y-3">
              {capitalCalls.items.map((call: any) => (
                <TransactionRow
                  key={call.id}
                  date={call.call_date}
                  type={call.call_type}
                  amount={call.amount}
                  isNegative
                />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No capital calls found</p>
          )}
        </div>

        {/* Distributions */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Distributions</h2>
          {distributions && distributions.items.length > 0 ? (
            <div className="space-y-3">
              {distributions.items.map((dist: any) => (
                <TransactionRow
                  key={dist.id}
                  date={dist.distribution_date}
                  type={dist.distribution_type}
                  amount={dist.amount}
                  isRecallable={dist.is_recallable}
                />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No distributions found</p>
          )}
        </div>
      </div>
    </div>
  )
}

// Helper function to prepare cash flow data for chart
function prepareCashFlowData(capitalCalls: any[], distributions: any[]) {
  const dataMap = new Map<string, { capitalCalls: number; distributions: number }>()

  // Aggregate capital calls
  capitalCalls.forEach(call => {
    const date = formatDate(call.call_date)
    if (!dataMap.has(date)) {
      dataMap.set(date, { capitalCalls: 0, distributions: 0 })
    }
    dataMap.get(date)!.capitalCalls -= call.amount // Negative for outflow
  })

  // Aggregate distributions
  distributions.forEach(dist => {
    const date = formatDate(dist.distribution_date)
    if (!dataMap.has(date)) {
      dataMap.set(date, { capitalCalls: 0, distributions: 0 })
    }
    dataMap.get(date)!.distributions += dist.amount // Positive for inflow
  })

  // Convert to array and sort by date
  return Array.from(dataMap.entries())
    .map(([date, data]) => ({ date, ...data }))
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    .slice(0, 20) // Limit to 20 most recent
}

// Helper function to prepare cumulative data
function prepareCumulativeData(capitalCalls: any[], distributions: any[]) {
  // Combine and sort all transactions
  const allTransactions = [
    ...capitalCalls.map(call => ({
      date: new Date(call.call_date),
      amount: -call.amount,
      type: 'call'
    })),
    ...distributions.map(dist => ({
      date: new Date(dist.distribution_date),
      amount: dist.amount,
      type: 'distribution'
    }))
  ].sort((a, b) => a.date.getTime() - b.date.getTime())

  let cumulativeCalls = 0
  let cumulativeDistributions = 0

  return allTransactions.map(transaction => {
    if (transaction.type === 'call') {
      cumulativeCalls += Math.abs(transaction.amount)
    } else {
      cumulativeDistributions += transaction.amount
    }

    return {
      date: formatDate(transaction.date.toISOString()),
      cumulativeCalls,
      cumulativeDistributions,
      netCashFlow: cumulativeDistributions - cumulativeCalls
    }
  }).slice(-20) // Show last 20 data points
}

function MetricCard({ title, value, description, icon, color }: {
  title: string
  value: string
  description: string
  icon: React.ReactNode
  color: 'blue' | 'green' | 'purple' | 'orange'
}) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    purple: 'bg-purple-100 text-purple-600',
    orange: 'bg-orange-100 text-orange-600',
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className={`w-12 h-12 rounded-lg ${colorClasses[color]} flex items-center justify-center mb-4`}>
        {icon}
      </div>
      <h3 className="text-sm font-medium text-gray-600 mb-1">{title}</h3>
      <p className="text-2xl font-bold text-gray-900 mb-1">{value}</p>
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  )
}

function TransactionRow({ date, type, amount, isNegative, isRecallable }: {
  date: string
  type: string
  amount: number
  isNegative?: boolean
  isRecallable?: boolean
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
      <div className="flex-1">
        <p className="text-sm font-medium text-gray-900">{type}</p>
        <div className="flex items-center space-x-2 mt-1">
          <Calendar className="w-3 h-3 text-gray-400" />
          <p className="text-xs text-gray-500">{formatDate(date)}</p>
          {isRecallable && (
            <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-0.5 rounded">
              Recallable
            </span>
          )}
        </div>
      </div>
      <p className={`text-sm font-semibold ${isNegative ? 'text-red-600' : 'text-green-600'}`}>
        {isNegative ? '-' : '+'}{formatCurrency(Math.abs(amount))}
      </p>
    </div>
  )
}
