'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fundApi } from '@/lib/api'
import { formatCurrency, formatPercentage } from '@/lib/utils'
import { Loader2, TrendingUp, TrendingDown, Award, BarChart } from 'lucide-react'
import axios from 'axios'

export default function ComparePage() {
  const [selectedFunds, setSelectedFunds] = useState<number[]>([])
  const [comparisonData, setComparisonData] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const { data: funds } = useQuery({
    queryKey: ['funds'],
    queryFn: () => fundApi.list()
  })

  const handleFundToggle = (fundId: number) => {
    setSelectedFunds(prev => {
      if (prev.includes(fundId)) {
        return prev.filter(id => id !== fundId)
      } else if (prev.length < 5) {
        return [...prev, fundId]
      }
      return prev
    })
  }

  const handleCompare = async () => {
    if (selectedFunds.length < 2) return

    setLoading(true)
    try {
      const queryString = selectedFunds.map(id => `fund_ids=${id}`).join('&')
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/compare?${queryString}`
      )
      setComparisonData(response.data)
    } catch (error) {
      console.error('Comparison failed:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Compare Funds</h1>
        <p className="text-gray-600">
          Select 2-5 funds to compare their performance metrics side-by-side
        </p>
      </div>

      {/* Fund Selection */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">
          Select Funds ({selectedFunds.length}/5)
        </h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {funds?.map((fund: any) => (
            <label
              key={fund.id}
              className={`
                relative flex items-center p-4 border-2 rounded-lg cursor-pointer transition
                ${selectedFunds.includes(fund.id)
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
                }
              `}
            >
              <input
                type="checkbox"
                checked={selectedFunds.includes(fund.id)}
                onChange={() => handleFundToggle(fund.id)}
                disabled={!selectedFunds.includes(fund.id) && selectedFunds.length >= 5}
                className="mr-3"
              />
              <div className="flex-1">
                <p className="font-semibold text-gray-900">{fund.name}</p>
                {fund.gp_name && (
                  <p className="text-sm text-gray-600">GP: {fund.gp_name}</p>
                )}
                {fund.vintage_year && (
                  <p className="text-xs text-gray-500">Vintage: {fund.vintage_year}</p>
                )}
              </div>
            </label>
          ))}
        </div>

        <button
          onClick={handleCompare}
          disabled={selectedFunds.length < 2 || loading}
          className="mt-6 w-full md:w-auto px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Comparing...</span>
            </>
          ) : (
            <>
              <BarChart className="w-5 h-5" />
              <span>Compare Selected Funds</span>
            </>
          )}
        </button>
      </div>

      {/* Comparison Results */}
      {comparisonData && (
        <div className="space-y-8">
          {/* Insights */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6 border border-blue-200">
              <div className="flex items-center space-x-3 mb-3">
                <Award className="w-6 h-6 text-blue-600" />
                <h3 className="text-lg font-semibold text-blue-900">Best DPI</h3>
              </div>
              <p className="text-2xl font-bold text-blue-900 mb-1">
                {comparisonData.insights.best_dpi.fund_name}
              </p>
              <p className="text-blue-700">
                DPI: {comparisonData.insights.best_dpi.dpi.toFixed(2)}x
              </p>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6 border border-green-200">
              <div className="flex items-center space-x-3 mb-3">
                <Award className="w-6 h-6 text-green-600" />
                <h3 className="text-lg font-semibold text-green-900">Best IRR</h3>
              </div>
              <p className="text-2xl font-bold text-green-900 mb-1">
                {comparisonData.insights.best_irr.fund_name}
              </p>
              <p className="text-green-700">
                IRR: {formatPercentage(comparisonData.insights.best_irr.irr)}
              </p>
            </div>
          </div>

          {/* Comparison Table */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="px-6 py-4 bg-gray-50 border-b">
              <h2 className="text-xl font-semibold">Detailed Comparison</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Fund Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      GP
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      DPI
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      IRR
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Paid-In Capital
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Distributions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {comparisonData.funds.map((fund: any) => (
                    <tr key={fund.fund_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">
                          {fund.fund_name}
                        </div>
                        <div className="text-xs text-gray-500">
                          {fund.vintage_year && `Vintage ${fund.vintage_year}`}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                        {fund.gp_name || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-semibold">
                            {fund.metrics.dpi.toFixed(2)}x
                          </span>
                          {fund.metrics.dpi > comparisonData.averages.dpi ? (
                            <TrendingUp className="w-4 h-4 text-green-600" />
                          ) : (
                            <TrendingDown className="w-4 h-4 text-red-600" />
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-semibold">
                            {formatPercentage(fund.metrics.irr)}
                          </span>
                          {fund.metrics.irr > comparisonData.averages.irr ? (
                            <TrendingUp className="w-4 h-4 text-green-600" />
                          ) : (
                            <TrendingDown className="w-4 h-4 text-red-600" />
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                        {formatCurrency(fund.metrics.pic)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                        {formatCurrency(fund.metrics.total_distributions)}
                      </td>
                    </tr>
                  ))}

                  {/* Averages Row */}
                  <tr className="bg-blue-50 font-semibold">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-900">
                      Average
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-900">
                      -
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-900">
                      {comparisonData.averages.dpi.toFixed(2)}x
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-900">
                      {formatPercentage(comparisonData.averages.irr)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-900">
                      {formatCurrency(comparisonData.averages.pic)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-900">
                      -
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
